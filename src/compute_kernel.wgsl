struct CpuData {
    speed: vec3<f32>,
}

struct InstanceInput {
    model_matrix_0: vec4<f32>,
    model_matrix_1: vec4<f32>,
    model_matrix_2: vec4<f32>,
    model_matrix_3: vec4<f32>,
    color: vec4<f32>,
};

@group(0) @binding(0)
var<storage, read_write> cpu_data: array<CpuData>;

@group(0) @binding(1)
var<storage, read_write> instances: array<InstanceInput>;

@compute @workgroup_size(1,1,1)
fn main(@builtin(global_invocation_id) GlobalInvocationID : vec3<u32>) {
    let index = GlobalInvocationID.x + (GlobalInvocationID.y * u32(10000));
    var instance: InstanceInput = instances[index];
    let model_matrix = mat4x4<f32>(
        instance.model_matrix_0,
        instance.model_matrix_1,
        instance.model_matrix_2,
        instance.model_matrix_3,
    );
    let speed = cpu_data[index].speed;
    instances[index].model_matrix_3 = instance.model_matrix_3 + vec4<f32>(speed.x, speed.y, speed.z, 0.0);
}
