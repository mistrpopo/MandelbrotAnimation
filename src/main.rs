#![allow(unused_variables)]

#[macro_use]
extern crate vulkano_shader_derive;
extern crate vulkano;
extern crate image;

use vulkano::instance::Instance;
use vulkano::instance::InstanceExtensions;
use vulkano::instance::PhysicalDevice;

use vulkano::device::Device;
use vulkano::device::Queue;
use vulkano::device::DeviceExtensions;
use vulkano::instance::Features;

use vulkano::buffer::BufferUsage;
use vulkano::buffer::CpuAccessibleBuffer;

use vulkano::command_buffer::AutoCommandBufferBuilder;
use vulkano::command_buffer::CommandBuffer;
use vulkano::sync::GpuFuture;

use std::sync::Arc;

use vulkano::pipeline::ComputePipeline;
use vulkano::descriptor::descriptor_set::PersistentDescriptorSet;

use vulkano::format::Format;
use vulkano::image::Dimensions;
use vulkano::image::StorageImage;

use image::ImageBuffer;
use image::Rgba;

mod mandelbrot_shader {
    #[derive(VulkanoShader)]
    #[ty = "compute"]
    #[src = "
#version 450

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(set = 0, binding = 0, rgba8) uniform writeonly image2D img;
layout(set = 0, binding = 1) buffer Data {
    float init;
} buf;

void main() {
    vec2 norm_coordinates = (gl_GlobalInvocationID.xy + vec2(0.5)) / vec2(imageSize(img));
    vec2 c = (norm_coordinates - vec2(0.5)) * 2.0 - vec2(1.0, 0.0);

    vec2 z = vec2(buf.init, buf.init);
    float i;
    for (i = 0.0; i < 1.0; i += 0.005) {
        z = vec2(
            z.x * z.x - z.y * z.y + c.x, //unfolded z**2 + c
            z.y * z.x + z.x * z.y + c.y
        );

        if (length(z) > 4.0) { //length computes the vector norm sqrt(z.x**2 + z.y**2)
            break;
        }
    }

    vec4 to_write = vec4(vec3(i), 1.0);
    imageStore(img, ivec2(gl_GlobalInvocationID.xy), to_write);
}//"
	]
	#[allow(dead_code)]
	struct Dummy;
}

fn main() {
	let (device, queue) = create_vulkan();
	let image = create_image(device.clone(), queue.clone());
	for wtf_rust_ranges in 0..500 {
		let init: f32 = (wtf_rust_ranges as f32) / 500.0;
		generate_mandelbrot(init, image.clone(), device.clone(), queue.clone());
	}
}

fn create_vulkan() -> (Arc<Device>, Arc<Queue>) {
	let instance = Instance::new(None, &InstanceExtensions::none(), None).expect("failed to create instance");
	let physical = PhysicalDevice::enumerate(&instance).next().expect("no device available");
	for family in physical.queue_families() {
		println!("Found a queue family with {:?} queue(s)", family.queues_count());
	}
	let queue_family = physical.queue_families()
		.find(|&q| q.supports_graphics())
		.expect("couldn't find a graphical queue family");
	let (device, mut queues) = {
	Device::new(physical, &Features::none(), &DeviceExtensions::none(),
		[(queue_family, 0.5)].iter().cloned()).expect("failed to create device")
	};
	let queue = queues.next().unwrap();
	(device, queue)
}

fn create_image(device: Arc<Device>,queue: Arc<Queue>) -> Arc<StorageImage<Format>> {
	let image = StorageImage::new(device.clone(), Dimensions::Dim2d { width: 1024, height: 1024 }, 
		Format::R8G8B8A8Unorm, Some(queue.family())).unwrap();
	image
}

fn generate_mandelbrot(init: f32, image: Arc<StorageImage<Format>>, device: Arc<Device>, queue: Arc<Queue>) {
	let shader = mandelbrot_shader::Shader::load(device.clone()).expect("failed to create shader module");
	let compute_pipeline = Arc::new(ComputePipeline::new(device.clone(), &shader.main_entry_point(), &()).expect("failed to create compute pipeline"));

	let init_buf = CpuAccessibleBuffer::from_data(device.clone(), BufferUsage::all(), init).expect("failed to create buffer");


	let set = Arc::new(PersistentDescriptorSet::start(compute_pipeline.clone(), 0)
	.add_image(image.clone()).unwrap()
	.add_buffer(init_buf.clone()).unwrap()
	.build().unwrap());

	let buf = CpuAccessibleBuffer::from_iter(device.clone(), BufferUsage::all(),
	(0 .. 1024 * 1024 * 4).map(|_| 0u8))
	.expect("failed to create buffer");

	let command_buffer = AutoCommandBufferBuilder::new(device.clone(), queue.family()).unwrap()
	.dispatch([1024 / 8, 1024 / 8, 1], compute_pipeline.clone(), set.clone(), ()).unwrap()
	.copy_image_to_buffer(image.clone(), buf.clone()).unwrap()
	.build().unwrap();

	let finished = command_buffer.execute(queue.clone()).unwrap();
	finished.then_signal_fence_and_flush().unwrap().wait(None).unwrap();

	let buffer_content = buf.read().unwrap();
	println!("Store buffer into image");
	let image = ImageBuffer::<Rgba<u8>, _>::from_raw(1024, 1024, &buffer_content[..]).unwrap();
	let path = format!("image_{:.3}.png", init);
	println!("Saving image to {:?}", path);
	image.save(path.clone()).unwrap();
	println!("Image saved to {:?}", path);
}