struct Genome {
    w: array<f32, 246> // V43 ALPS RNN Memory topology (128 + 16 + 96 + 6)
}

struct Uniforms {
    generation: u32,
    num_map_elites: u32,
    target_fitness: f32,
    global_best_f: f32
}

@group(0) @binding(0) var<storage, read> genomes: array<Genome>;
@group(0) @binding(1) var<storage, read_write> fitnesses: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> signals: array<f32>; 
@group(0) @binding(3) var<storage, read> apex_targets: array<i32>;
@group(0) @binding(4) var<uniform> uniforms: Uniforms;
@group(0) @binding(5) var<storage, read_write> next_genomes: array<Genome>;
@group(0) @binding(6) var<storage, read> map_genomes: array<Genome>;

const N_A: u32 = 16u;
const ML: u32 = 24u;
const L: u32 = 1500u;
const METABOLIC_RATE: f32 = 0.00001;  // Sustainable for hourly ticks over 1500 hours



// V44: GPU-Native Multiplicative Hash RNG (Knuth / Wang variant)
fn hash(seed: u32) -> u32 {
    var state = seed;
    state = state ^ 2747636419u;
    state = state * 2654435769u;
    state = state ^ (state >> 16u);
    state = state * 2654435769u;
    state = state ^ (state >> 16u);
    state = state * 2654435769u;
    return state;
}

fn randomFloat(seed: ptr<function, u32>) -> f32 {
    *seed = hash(*seed);
    return f32(*seed) / 4294967295.0;
}

fn boxMuller(seed: ptr<function, u32>) -> f32 {
    let u1 = max(randomFloat(seed), 1e-9);
    let u2 = randomFloat(seed);
    return sqrt(-2.0 * log(u1)) * cos(6.28318530718 * u2);
}

// V44: GPU Tournament Selection (Replaces Javascript Array Sorting)
fn tournament_selection(island_start: u32, island_size: u32, seed: ptr<function, u32>) -> u32 {
    let p1 = island_start + u32(randomFloat(seed) * f32(island_size));
    let p2 = island_start + u32(randomFloat(seed) * f32(island_size));
    let p3 = island_start + u32(randomFloat(seed) * f32(island_size));
    
    var best = p1;
    var best_f = fitnesses[p1].x;
    if (fitnesses[p2].x > best_f) { best = p2; best_f = fitnesses[p2].x; }
    if (fitnesses[p3].x > best_f) { best = p3; best_f = fitnesses[p3].x; }
    return best;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= 10000u) { return; }
    
    // Copy genome to fast local memory
    var local_g: array<f32, 246>;
    for (var k = 0u; k < 246u; k++) {
        local_g[k] = genomes[idx].w[k];
    }
    
    // Simulation variables
    var pv = 1.0;
    var pp = 1.0;
    var hours_since_rest = 0.0;
    var pw: array<f32, 16>;
    
    var rets: array<f32, 5000>;
    
    // V43 Recurrent Neural Network (RNN) State
    var mem0 = 0.0;
    var mem1 = 0.0;
    
    // V42 MAP-Elites Behavioral Tracking
    var num_trades = 0.0;
    var peak_lev = 1.0;
    
    // The dataset we pass starts exactly at `ts - ML`, so h=0 represents the start of the evaluation window.
    // The total length of P buffer is `L + ML + 1`. (We need ML history before h=0).
    
    for (var h = 0u; h < L; h++) {
        let sig_base = h * 36u;
        
        let dd = select((pp - pv) / pp, 0.0, pp <= 0.0);
        
        var fir_pnl = max(0.0, (pv/pp) - 1.0) * 10.0;
        if (fir_pnl > 1.0) { fir_pnl = 1.0; }
        
        var drw_dwn = dd * 5.0;
        if (drw_dwn > 1.0) { drw_dwn = 1.0; }
        
        var hrs_rst = hours_since_rest / 168.0;
        if (hrs_rst > 1.0) { hrs_rst = 1.0; }
        
        var X = array<f32, 8>(
            signals[sig_base + 0u], // btc_vol * 10.0
            drw_dwn,
            hrs_rst,
            fir_pnl,
            signals[sig_base + 1u], // clamped global_mom
            signals[sig_base + 2u], // clamped top_mom
            mem0,                   // V43 RNN Memory In
            mem1
        );
        
        let emotions = nn_forward(X, &local_g);
        // V46: No more sleep. emotions[0] now controls CAUTION (scales max allocation)
        let caution = emotions[0]; // 0=aggressive, 1=conservative
        let lev_base = 1.0 + (emotions[1] * 2.0); // V46: 1x to 3x leverage (survivable)
        let maxw_base = (1.0 - caution * 0.8) * (0.2 + (emotions[3] * 0.8)); // caution shrinks allocation
        mem0 = emotions[4]; // V43 RNN Memory Recurse
        mem1 = emotions[5];
        
        // V46: ALWAYS ACTIVE — no sleep branch, network must learn to allocate wisely
        num_trades += 1.0;
        if (lev_base > peak_lev) { peak_lev = lev_base; }
        // (Removed unused bubble sort)
        // Allocate raw
        var tw = 0.0;
        var w: array<f32, 16>;
        for (var a = 0u; a < N_A; a++) {
            let sc_a = signals[sig_base + 4u + a];
            if (sc_a > 0.0) {
                w[a] = sc_a;
                tw += sc_a;
            }
        }
        if (tw > 0.0) {
            for (var a = 0u; a < N_A; a++) { w[a] /= tw; }
        }
        // Cap
        var tw2 = 0.0;
        for (var a = 0u; a < N_A; a++) {
            if (w[a] > maxw_base) { w[a] = maxw_base; }
            tw2 += w[a];
        }
        if (tw2 > 0.0) {
            for (var a = 0u; a < N_A; a++) { w[a] /= tw2; }
        }
        
        // HYSTERESIS JITTER FILTER (Local Optimum Fix)
        // Neural networks are continuous; outputs jitter by 0.1% every second.
        // At 50x leverage, re-balancing 0.1% incurs massive fees and bleeds the portfolio to zero in minutes.
        // We simulate "sticky" orders: if the target weight is within 5% of the current holding, DO NOTHING.
        for (var a = 0u; a < N_A; a++) {
            if (abs(w[a] - pw[a]) < 0.05) {
                w[a] = pw[a];
            }
        }
        
        // Return calculation
        var rr = 0.0;
        for (var a = 0u; a < N_A; a++) {
            if (w[a] > 0.0) {
                let ra = signals[sig_base + 20u + a];
                rr += w[a] * ra;
            }
        }
        
        var dr = rr * lev_base;
        for (var a = 0u; a < N_A; a++) {
            let diff = abs(w[a] - pw[a]);
            
            // V40 REALITY: MARKET IMPACT SPREAD (Liquidity Wall)
            // If the Swarm allocates >500% of collateral ($500) into a memecoin, orderbook slippage increases exponentially
            var liquidity_penalty = 0.0;
            if (lev_base * w[a] > 5.0) { 
                liquidity_penalty = (lev_base * w[a] - 5.0) * 0.0005; 
            }
            
            // V42 CO-EVOLUTION: Shadow Apex Predation
            // If the young genome tries to buy the same asset the Apex is holding, it gets front-run
            let apex_target_idx = apex_targets[h];
            if (apex_target_idx == i32(a) && w[a] > pw[a]) {
                liquidity_penalty += 0.0020; // 0.20% instant front-running penalty
            }
            
            // V41 REALITY: Asymmetric Maker/Taker Exits
            // Buying (w > pw) means taking liquidity: 0.1% Fee + 0.05% Slippage = 0.15%
            // Selling (w < pw) means making liquidity via Limit Orders: 0.05% Fee = 0.05%
            var friction = 0.0004;  // V46: Realistic taker fee for <$500 positions
            if (w[a] < pw[a]) { friction = 0.0002; }  // Maker limit order
            
            // TOTAL FRICTION
            let total_friction = friction + liquidity_penalty; 
            dr -= diff * lev_base * total_friction; 
        }
        
        dr -= METABOLIC_RATE; // Living tax
        rets[h] = dr;
        pv *= (1.0 + dr);
        
        // V40 REALITY: MAINTENANCE MARGIN (Instant Liquidation)
        if (pv <= 0.05) { // Liquidation at 95% loss
            pv = 0.0001;
            for(var k = h; k < L; k++) { rets[k] = -1.0; } // Flatline the genome
            break;
        }
        
        if (pv > pp) { pp = pv; }
        for (var a = 0u; a < N_A; a++) { pw[a] = w[a]; }
    }
    
    // Evaluate fitness
    var tr = 1.0;
    var mdd = 0.0;
    var pk = 1.0;
    var p = 1.0;
    var mean = 0.0;
    
    for (var h = 0u; h < L; h++) {
        tr *= (1.0 + rets[h]);
        p *= (1.0 + rets[h]);
        if (p > pk) { pk = p; }
        let current_dd = (pk - p) / pk;
        if (current_dd > mdd) { mdd = current_dd; }
        mean += rets[h];
    }
    
    mean /= f32(L);
    
    var std_dev = 0.0;
    for (var h = 0u; h < L; h++) {
        std_dev += (rets[h] - mean) * (rets[h] - mean);
    }
    std_dev = sqrt(std_dev / f32(L)) + 0.00001;
    
    let sh = (mean / std_dev) * sqrt(24.0 * 365.0); // Standardized metric base
    
    // V46: Relaxed MDD for hourly bear market data
    if (mdd > 0.85) {
        fitnesses[idx] = vec4<f32>(-9999.0, num_trades, peak_lev, 0.0);
        return;
    }
    
    // V46: Pure total return fitness (no activity gate needed — network is always active)
    fitnesses[idx] = vec4<f32>(tr, num_trades, peak_lev, 0.0);
}

@compute @workgroup_size(64)
fn evolve(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= 10000u) { return; }
    
    // Seed using unique genome ID and generation count
    var seed = idx + (uniforms.generation * 10000u) + 12345u;
    
    let ISLAND_COUNT = 10u;
    let ISLAND_SIZE = 1000u;
    let island = idx / ISLAND_SIZE;
    let island_start = island * ISLAND_SIZE;
    let local_idx = idx % ISLAND_SIZE;
    
    // Instead of exhaustive sweep (10,000 * 1000 = 10 million reads),
    // use a fast Elite Sweep (Check 20 random indices)
    var elite_idx = island_start;
    var elite_f = fitnesses[island_start].x;
    for (var i = 0u; i < 20u; i++) {
        let cand = island_start + u32(randomFloat(&seed) * f32(ISLAND_SIZE));
        if (fitnesses[cand].x > elite_f) {
            elite_f = fitnesses[cand].x;
            elite_idx = cand;
        }
    }
    
    // V44: Sub-Populations (Nursery vs Veterans)
    let is_nursery_wipe = (uniforms.generation > 0u && uniforms.generation % 100u == 0u && island == 0u);
    let is_extinction = (elite_f < 0.0);
    
    if (is_nursery_wipe) {
        for (var k = 0u; k < 246u; k++) {
            next_genomes[idx].w[k] = boxMuller(&seed) * 1.5;
        }
        return;
    }
    
    if (local_idx == 0u) { // Slot 0 is strictly reserved for the Elite
        for (var k = 0u; k < 246u; k++) {
            next_genomes[idx].w[k] = genomes[elite_idx].w[k];
        }
        return;
    }
    
    if (is_extinction && local_idx > u32(f32(ISLAND_SIZE) * 0.2)) {
        if (local_idx > u32(f32(ISLAND_SIZE) * 0.95)) {
            for (var k = 0u; k < 246u; k++) {
                next_genomes[idx].w[k] = boxMuller(&seed) * 1.5;
            }
        } else {
            for (var k = 0u; k < 246u; k++) {
                next_genomes[idx].w[k] = genomes[elite_idx].w[k] + (boxMuller(&seed) * 2.0);
            }
        }
        return;
    }
    
    // Select parents via Tournament
    let p1 = tournament_selection(island_start, ISLAND_SIZE, &seed);
    let p2 = tournament_selection(island_start, ISLAND_SIZE, &seed);
    
    let is_map = randomFloat(&seed) < 0.1 && uniforms.num_map_elites > 0u;
    var mp_idx = 0u;
    if (is_map) {
        mp_idx = u32(randomFloat(&seed) * f32(uniforms.num_map_elites));
    }
    
    let distance_to_target = max(0.1, (uniforms.target_fitness - uniforms.global_best_f) / uniforms.target_fitness);
    let mutation_rate = 0.5 * min(1.0, distance_to_target + 0.1);
    
    for (var k = 0u; k < 246u; k++) {
        let w1 = genomes[p1].w[k];
        var w2 = genomes[p2].w[k];
        if (is_map) { w2 = map_genomes[mp_idx].w[k]; }
        
        var w = w2;
        if (randomFloat(&seed) < 0.5) { w = w1; }
        
        let mut_chance = randomFloat(&seed);
        if (mut_chance < 0.05) {
            w += (boxMuller(&seed) * mutation_rate * 2.0);
        } else if (mut_chance < 0.20) {
            w += (boxMuller(&seed) * mutation_rate * 0.1);
        }
        
        next_genomes[idx].w[k] = w;
    }
}
