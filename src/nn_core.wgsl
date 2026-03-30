// Core V44 Neural Net Math (Shared by Evolution and Execution)
fn nn_forward(X: array<f32, 8>, g: ptr<function, array<f32, 246>>) -> array<f32, 6> {
    var H: array<f32, 16>;
    for (var i = 0u; i < 16u; i++) {
        var acc = (*g)[128u + i]; // B1
        for (var j = 0u; j < 8u; j++) {
            acc += X[j] * (*g)[j * 16u + i]; // W1
        }
        var ext = exp(clamp(2.0 * acc, -20.0, 20.0));
        H[i] = (ext - 1.0) / (ext + 1.0); 
    }
    
    var Out: array<f32, 6>;
    for (var i = 0u; i < 6u; i++) {
        var acc = (*g)[240u + i]; // B2
        for (var j = 0u; j < 16u; j++) {
            acc += H[j] * (*g)[144u + j * 6u + i]; // W2
        }
        Out[i] = 1.0 / (1.0 + exp(clamp(-acc, -20.0, 20.0))); 
    }
    return Out;
}
