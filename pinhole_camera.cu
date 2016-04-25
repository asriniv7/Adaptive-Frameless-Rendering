
/*
 * Copyright (c) 2008 - 2009 NVIDIA Corporation.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property and proprietary
 * rights in and to this software, related documentation and any modifications thereto.
 * Any use, reproduction, disclosure or distribution of this software and related
 * documentation without an express license agreement from NVIDIA Corporation is strictly
 * prohibited.
 *
 * TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THIS SOFTWARE IS PROVIDED *AS IS*
 * AND NVIDIA AND ITS SUPPLIERS DISCLAIM ALL WARRANTIES, EITHER EXPRESS OR IMPLIED,
 * INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
 * PARTICULAR PURPOSE.  IN NO EVENT SHALL NVIDIA OR ITS SUPPLIERS BE LIABLE FOR ANY
 * SPECIAL, INCIDENTAL, INDIRECT, OR CONSEQUENTIAL DAMAGES WHATSOEVER (INCLUDING, WITHOUT
 * LIMITATION, DAMAGES FOR LOSS OF BUSINESS PROFITS, BUSINESS INTERRUPTION, LOSS OF
 * BUSINESS INFORMATION, OR ANY OTHER PECUNIARY LOSS) ARISING OUT OF THE USE OF OR
 * INABILITY TO USE THIS SOFTWARE, EVEN IF NVIDIA HAS BEEN ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGES
 */

#include <optix.h>
#include <optix_world.h>
#include <optixu\optixu_math_namespace.h>
#include "helpers.h"
#include"random.h"
#include<curand.h>
#include"Tile.h"
#include<stdlib.h> 
#include<Windows.h> 

//#include<glm\glm\mat4x4.hpp>
//#include<glm\glm\glm.hpp>
//#include<glm\glm\gtc\matrix_transform.hpp>

#include<glm-0.9.7.1\glm\glm\mat4x4.hpp>
#include<glm-0.9.7.1\glm\glm\glm.hpp>
#include<glm-0.9.7.1\glm\glm\gtc\matrix_transform.hpp>

using namespace optix;

struct PerRayData_radiance
{
  float3 result;
  float  importance;
  int    depth;
  float3 hit_point;
};


rtBuffer<List, 2>				 deep_buffer; //deep buffer
//rtBuffer<Tile, 2> deep_buffer; 


rtDeclareVariable(float3,        eye, , );
rtDeclareVariable(float3,        U, , );
rtDeclareVariable(float3,        V, , );
rtDeclareVariable(float3,        W, , );
rtDeclareVariable(float3,        bad_color, , );
rtDeclareVariable(float,         scene_epsilon, , );
rtDeclareVariable(unsigned int,	 tile_size, , ); //Stores the size of the tile to which to shoot rays to
rtDeclareVariable(float,			 number_of_parent_tiles, , ); //Stores the number of parent tiles. 

rtBuffer<uchar4, 2>              output_buffer;
rtBuffer<uchar4, 2>				 color_buffer; //Stores the color value of the pixel before tranparency
rtBuffer<uchar4, 2>				 tiled_buffer;
rtBuffer<float3, 2>				 wixi_buffer; //Stores the weighted sum of colors
rtBuffer<float, 2>				 wi_buffer; //Stores the sum of weights 
rtBuffer<float3, 2>				 G_buffer;
rtBuffer<int , 2>				 stencil_buffer; //1 if should not be reprojected. 0 if it should.
rtBuffer<int, 2>				 busy_buffer; //1 if it's being used. 0 if free. 
rtBuffer<float4, 2>              variance_sum_buffer;//Stores the variance 
rtBuffer<float4, 2>              variance_sum2_buffer;//Stores variance squared
rtBuffer<float3, 2>				 temp_buffer;//Stores the previous pixel's color				 
rtBuffer<unsigned int, 2>        rnd_seeds;
rtBuffer<float, 1>				 variance_buffer; //Stores the variance computed by the GPU Feb 9th.
rtBuffer<float, 1>				 parent_variance_buffer; //Stores the variance of the parent tiles. 
rtBuffer<float3, 1>				 gradient_buffer; //Stores the average x gradient of all pixels in the tile

rtBuffer<uchar4, 2>				 gaussian_x_buffer; 
rtBuffer<uchar4, 2>				 gaussian_y_buffer;

rtDeclareVariable(rtObject,      top_object, , );
rtDeclareVariable(unsigned int,  radiance_ray_type, , );

rtDeclareVariable(uint2, launch_index, rtLaunchIndex, );
rtDeclareVariable(uint2, launch_dim,   rtLaunchDim, );
rtDeclareVariable(float, time_view_scale, , ) = 1e-6f;
rtDeclareVariable(uint2, rand_tile_index, , );//Index of the tile to which to shoot rays to

rtBuffer<uint2, 1>				leaf_tile_indices;//Stores the indices of the leaf tiles
rtBuffer<unsigned int, 1>		leaf_tile_sizes;//Stores the size of the corresponding leaf tile

rtBuffer<uint2, 1>              parent_tile_indices;
rtBuffer<unsigned int, 1>		parent_tile_sizes; 

#define PI 3.1415926

//#define TIME_VIEW

__device__ float gaussian_3[3] = {0.05908, 0.88183, 0.05908}; 
__device__ float gaussian_5[5] = {0.06136, 0.24477, 0.38774, 0.24477, 0.06136};
__device__ float gaussian_7[7] = {0.05960, 0.12645, 0.19855, 0.23071, 0.19855, 0.12645, 0.05960};

//Checks whether a pixel is centre of block. Just like shouldTrace.
static __device__ __inline__ bool isCentre(const uint2& index, unsigned int spacing)
{
	unsigned int half_spacing  = spacing >> 1;
  uint2        shifted_index = make_uint2( index.x + half_spacing, index.y + half_spacing ); 
  size_t2      screen        = output_buffer.size(); 
  return ( shifted_index.x % spacing == 0 && shifted_index.y % spacing == 0 ) ||
         ( index.x == screen.x-1 && screen.x % spacing <= half_spacing && shifted_index.y % spacing == 0 ) ||
         ( index.y == screen.y-1 && screen.y % spacing <= half_spacing && shifted_index.x % spacing == 0 );

  int f = 1.0;
  int* p = &f;
  atomicAdd(p, 1); 
  atomicExch(p, 1);
}

static __device__ __inline__ float3 trace( float2 screen_coord )
{
  size_t2 screen = output_buffer.size();
  float2 d = screen_coord / make_float2(screen) * 2.f - 1.f;
  float3 ray_origin = eye;
  float3 ray_direction = normalize(d.x*U + d.y*V + W);
  
  optix::Ray ray = optix::make_Ray(ray_origin, ray_direction, radiance_ray_type, scene_epsilon, RT_DEFAULT_MAX);

  PerRayData_radiance prd;
  prd.importance = 1.f;
  prd.depth = 0;

  rtTrace(top_object, ray, prd);

  //rtPrintf("Closest hit = %f %f %f \n", prd.hit_point.x, prd.hit_point.y, prd.hit_point.z);

  G_buffer[launch_index] = make_float3(prd.hit_point.x, prd.hit_point.y, prd.hit_point.z);

  return prd.result;
}

static __device__ __inline__ void shoot_afr()
{
	uint2 new_index = make_uint2(launch_index.x, launch_index.y);
	size_t2 screen = output_buffer.size();

	uint2 centre_pixel = make_uint2(leaf_tile_indices[new_index.x].y,
									leaf_tile_indices[new_index.x].x);

	
	volatile unsigned int random_seed = rnd_seeds[new_index];
	unsigned int seed = random_seed;
	float x_offset = rnd(seed);
	float y_offset = rnd(seed); 
	rnd_seeds[launch_index] = seed;
	
	unsigned int tile_size = leaf_tile_sizes[new_index.x];

	uint2 start_pixel;
	start_pixel.x = centre_pixel.x - tile_size/2u;
	start_pixel.y = centre_pixel.y - tile_size/2u;

	x_offset *= tile_size;
	y_offset *= tile_size;
	uint2 total_offset = make_uint2(x_offset, y_offset);
	uint2 rand_pixel = start_pixel + total_offset;

	//Centre pixel
	float3 centre_pixel_color = trace(make_float2(rand_pixel));
	output_buffer[rand_pixel] = make_color(centre_pixel_color);
	color_buffer[rand_pixel] = make_color(centre_pixel_color); 
	stencil_buffer[rand_pixel] = 1;
	//Top pixel
	uint2 top_pixel = rand_pixel + make_uint2(0u,1u);
	float3 top_pixel_color = trace(make_float2(top_pixel));
	output_buffer[top_pixel] = make_color(top_pixel_color); 
	color_buffer[top_pixel] = make_color(top_pixel_color); 
	
	//Right pixel
	uint2 right_pixel = rand_pixel + make_uint2(1u,0u);
	float3 right_pixel_color = trace(make_float2(right_pixel));
	output_buffer[right_pixel] = make_color(right_pixel_color);
	color_buffer[right_pixel] = make_color(right_pixel_color); 

	
	//Calculate luminances for the samples
	float lum_centre = luminance(centre_pixel_color);
	float lum_top = luminance(top_pixel_color);
	float lum_right = luminance(right_pixel_color);

	uchar4 old_color_char = deep_buffer[rand_pixel].return_sample_at_age(0);
	float3 old_color = make_float3(old_color_char.z, old_color_char.y, old_color_char.x)*make_float3(1.0f/255.99f);
	//rtPrintf("old _ color = %f %f %f \t %u %u\n", old_color.x, old_color.y, old_color.z, launch_index.x, launch_index.y); 
	float lum_temporal = luminance(old_color);

	clock_t time_stamp = deep_buffer[rand_pixel].get_sample_time();
	clock_t time_elapsed = clock() - time_stamp;
	deep_buffer[rand_pixel].set_sample_time();

	float g_x = abs(lum_centre - lum_right);
	float g_y = abs(lum_centre - lum_top);
	float g_t = (abs(lum_temporal - lum_centre)) / abs((float)time_elapsed/CLOCKS_PER_SEC);
	g_t /= time_view_scale; 

	//Add crosshair to deep buffer  
	//deep_buffer[rand_pixel].add_sample(make_color(centre_pixel_color), g_x, g_y, g_t);
	deep_buffer[rand_pixel].add_sample_simple(make_color(centre_pixel_color)); 

	//Use gradient to calculate extents
	//float3 gradient = deep_buffer[rand_pixel].get_gradients();
	float3 gradient = make_float3(g_x, g_y, g_t); 

	float3 extent;
	extent.x = extent.y = extent.z = 0.0f;
	float vs = 0.04;
	float rl = 100.0/ ((float(tile_size) * (float)tile_size) * (float)0.07); 
	vs = 1/rl; 

	if(gradient.x == 0.0f || gradient.y == 0.0f || gradient.z == 0.0f)
	{
		extent.x = extent.y = extent.z = 0.0f;
	}
	else
	{
		extent.x = pow((gradient.y * gradient.z * vs)/(gradient.x * gradient.x),0.33f);
		extent.y = pow((gradient.x * gradient.z * vs)/(gradient.y * gradient.y),0.33f);
		extent.z = pow((gradient.x * gradient.y * vs)/(gradient.z * gradient.z),0.33f);
	}

	deep_buffer[rand_pixel].set_extents(extent); 
	
}

static __device__ __inline__ void shoot_new()
{
	uint2 new_index = make_uint2(launch_index.x % 256u, launch_index.y % 16u); 

	size_t2 screen = output_buffer.size();
	volatile unsigned int seed  = rnd_seeds[ new_index ];
  unsigned int new_seed  = seed;
  float uu = rnd( new_seed );
  float vv = rnd(new_seed);
  uint2 random_index = make_uint2(uu*screen.x, vv*screen.y);

  rnd_seeds[ new_index ] = new_seed;

  //Shoot ray offsetting from centre pixel of leaf tiles

  //Get centre pixel associated with thread
  uint2 centre_pixel = make_uint2(leaf_tile_indices[new_index.x].y, 
								  leaf_tile_indices[new_index.x].x);

  //stencil_buffer[centre_pixel] = 1;

  //Launch X rays per thread
  for(int i = 0;i<50;i++)
  {
	  float x_offset = rnd(new_seed);
	  float y_offset = rnd(new_seed);
	  float x_direction = rnd(new_seed);
	  float y_direction = rnd(new_seed);

	  unsigned int tile_size = leaf_tile_sizes[new_index.x];

	  uint2 start_pixel;
	  start_pixel.x = centre_pixel.x - tile_size /2u;
	  start_pixel.y = centre_pixel.y - tile_size /2u;

	  x_offset *= tile_size;
	  y_offset *= tile_size;
	  uint2 total_offset = make_uint2(x_offset, y_offset);
	  uint2 rand_pixel = start_pixel + total_offset;

	  
	  //Right pixel
	  uint2 right_pixel = start_pixel + make_uint2(x_offset+1.0f, y_offset);
	  //Top pixel
	  uint2 top_pixel = start_pixel + make_uint2(x_offset, y_offset+1.0f);

	  //Now actually shoot the center ray
	  float2 d = (make_float2(rand_pixel))/make_float2(screen) * 2.f - 1.f;
	  float3 ray_origin = eye;
	  float3 ray_direction = normalize(d.x*U + d.y*V + W);

	  optix::Ray ray = optix::make_Ray(ray_origin, ray_direction, radiance_ray_type, scene_epsilon, RT_DEFAULT_MAX);

	  PerRayData_radiance prd;
	  prd.importance = 1.f;
	  prd.depth = 0;
	  
	  //Shoot right ray
	  float2 d_right = (make_float2(right_pixel))/make_float2(screen) * 2.f - 1.f;
	  float3 ray_dir_right = normalize(d_right.x*U + d_right.y*V + W);

	  optix::Ray ray_right = optix::make_Ray(ray_origin, ray_dir_right, radiance_ray_type, scene_epsilon, RT_DEFAULT_MAX);
	  PerRayData_radiance prd_right;
	  prd_right.importance = 1.f;
	  prd_right.depth = 0;

	  //Shoot top ray
	  float2 d_top = (make_float2(top_pixel))/make_float2(screen) * 2.f - 1.f;
	  float3 ray_dir_top = normalize(d_top.x*U + d_top.y*V + W);

	  optix::Ray ray_top = optix::make_Ray(ray_origin, ray_dir_top, radiance_ray_type, scene_epsilon, RT_DEFAULT_MAX);
	  PerRayData_radiance prd_top;
	  prd_top.importance = 1.f;
	  prd_top.depth = 0;
 
	  //Shoot delayed ray at centre pixel
	  PerRayData_radiance prd_delayed;
	  prd_delayed.importance = 1.f;
	  prd_delayed.depth = 0;
	  
	  //Shoot all rays together
	  
	  rtTrace(top_object, ray, prd);
	  rtTrace(top_object, ray_right, prd_right);
	  rtTrace(top_object, ray_top, prd_top);

	  float lum_centre = luminance(prd.result);
	  float lum_right = luminance(prd_right.result);
	  float lum_top = luminance(prd_top.result);

	  uchar4 old_color_char = deep_buffer[rand_pixel].return_sample_at_age(0);
	  float3 old_color = make_float3(old_color_char.z, old_color_char.y, old_color_char.x)*make_float3(1.0f/255.99f);
	  float lum_temporal = luminance(old_color); 
	  
	  //Add results to buffers
	  
		output_buffer[rand_pixel] = make_color(prd.result);
	    color_buffer[rand_pixel] = make_color(prd.result);
		//Write neighboring new samples too
		output_buffer[top_pixel] = make_color(prd_top.result);
		output_buffer[right_pixel] = make_color(prd_right.result);
		color_buffer[top_pixel] = make_color(prd_top.result);
		color_buffer[right_pixel] = make_color(prd_right.result); 
		//
	    //G_buffer[rand_pixel] = make_float3(prd.hit_point.x, prd.hit_point.y, prd.hit_point.z);
	    //stencil_buffer[rand_pixel] = 1;
		//atomicExch(&busy_buffer[rand_pixel], 0); //Set pixel to free. 

		
		//Calculate time elapsed
		clock_t time_stamp = deep_buffer[rand_pixel].get_sample_time();
		clock_t time_elapsed = clock() - time_stamp;
		//rtPrintf("%d clicks \t %f seconds since last sample \n", time_elapsed, ((float)time_elapsed/CLOCKS_PER_SEC)*time_view_scale);
		deep_buffer[rand_pixel].set_sample_time();

		//Calculate appropriate gradients
		/*
		float g_x = (abs(lum_left - lum_centre) + abs(lum_centre - lum_right))/2 ;
		float g_y = (abs(lum_bottom - lum_centre) + abs(lum_centre - lum_right))/2 ;
		float g_t = (abs(lum_temporal - lum_centre)) / abs((float)time_elapsed/CLOCKS_PER_SEC);
		g_t /= time_view_scale;
		*/

		
		float g_x = abs(lum_centre - lum_right);
		float g_y = abs(lum_centre - lum_top);
		float g_t = (abs(lum_temporal - lum_centre)) / abs((float)time_elapsed/CLOCKS_PER_SEC);
		g_t /= time_view_scale; 

		//Add crosshair to deep buffer  
		deep_buffer[rand_pixel].add_sample(make_color(prd.result), g_x, g_y, g_t); 
		//deep_buffer[rand_pixel].set_sample_time();

		float3 gradient = deep_buffer[rand_pixel].get_gradients();

		float3 extent;
		extent.x = extent.y = extent.z = 0.0f;
		float vs = 0.04;
		float rl = 100.0/ ((float(tile_size) * (float)tile_size) * (float)0.07); 
		vs = 1/rl; 
		//rtPrintf("%f \n", vs); 
		if(gradient.x == 0.0f || gradient.y == 0.0f || gradient.z == 0.0f)
		{
			extent.x = extent.y = extent.z = 0.0f;
		}
		else
		{
			extent.x = pow((gradient.y * gradient.z * vs)/(gradient.x * gradient.x),0.33f);
			extent.y = pow((gradient.x * gradient.z * vs)/(gradient.y * gradient.y),0.33f);
			extent.z = pow((gradient.x * gradient.y * vs)/(gradient.z * gradient.z),0.33f);
		}

		//rtPrintf("extent = %f %f %f \n", extent.x, extent.y, extent.z); 
		deep_buffer[rand_pixel].set_extents(extent); 
		
	 
	  

	  
  }

}


static __device__ __inline__ void shoot()
{
	size_t2 screen = output_buffer.size();
	//int r = rand() % 20; 
	//rtPrintf("%d \n", r);

	volatile unsigned int seed  = rnd_seeds[ launch_index ]; // volatile workaround for cuda 2.0 bug
	//rtPrintf("%u \n", launch_index.x);
  unsigned int new_seed  = seed;
  float uu = rnd( new_seed );
  float vv = rnd(new_seed);
  uint2 random_index = make_uint2(uu*screen.x, vv*screen.y);

  rnd_seeds[ launch_index ] = new_seed;

  //Shoot ray offsetting from centre pixel of leaf tiles
  //int rnd_tile_index = rnd(new_seed) * 256; 

  //rtPrintf(" Tile = %d \n", rnd_tile_index);

  //Get centre pixel associated with thread
  uint2 centre_pixel = make_uint2(leaf_tile_indices[launch_index.x].y, 
								  leaf_tile_indices[launch_index.x].x);
  
  //rtPrintf("Tile = %u \t %u %u\n", launch_index.x, centre_pixel.x, centre_pixel.y);

  stencil_buffer[centre_pixel] = 1;

  //Launch X rays per thread
  for(int i = 0;i<100;i++)
  {
	  float x_offset = rnd(new_seed);
	  float y_offset = rnd(new_seed);
	  float x_direction = rnd(new_seed);
	  float y_direction = rnd(new_seed);

	  unsigned int tile_size = leaf_tile_sizes[launch_index.x];

	  uint2 start_pixel;
	  start_pixel.x = centre_pixel.x - tile_size /2u;
	  start_pixel.y = centre_pixel.y - tile_size /2u;

	  x_offset *= tile_size;
	  y_offset *= tile_size;
	  uint2 total_offset = make_uint2(x_offset, y_offset);
	  uint2 rand_pixel = start_pixel + total_offset;

	  //Left pixel
	  uint2 left_pixel = start_pixel + make_uint2(x_offset-1.0f, y_offset);
	  //Right pixel
	  uint2 right_pixel = start_pixel + make_uint2(x_offset+1.0f, y_offset);
	  //Top pixel
	  uint2 top_pixel = start_pixel + make_uint2(x_offset, y_offset+1.0f);
	  //
	  uint2 bottom_pixel = start_pixel + make_uint2(x_offset, y_offset-1.0f); 

	  //Now actually shoot the center ray
	  float2 d = (make_float2(rand_pixel))/make_float2(screen) * 2.f - 1.f;
	  float3 ray_origin = eye;
	  float3 ray_direction = normalize(d.x*U + d.y*V + W);

	  optix::Ray ray = optix::make_Ray(ray_origin, ray_direction, radiance_ray_type, scene_epsilon, RT_DEFAULT_MAX);

	  PerRayData_radiance prd;
	  prd.importance = 1.f;
	  prd.depth = 0;

	  //rtTrace(top_object, ray, prd);
	  //float lum_centre = luminance(prd.result); 

	  //Now shoot the left ray
	  float2 d_left = (make_float2(left_pixel))/make_float2(screen) * 2.f - 1.f;
	  float3 ray_dir_left = normalize(d_left.x*U + d_left.y*V + W);

	  optix::Ray ray_left = optix::make_Ray(ray_origin, ray_dir_left, radiance_ray_type, scene_epsilon, RT_DEFAULT_MAX);
	  PerRayData_radiance prd_left;
	  prd_left.importance = 1.f;
	  prd_left.depth = 0;

	  //rtTrace(top_object, ray_left, prd_left); 
	  //float lum_left = luminance(prd_left.result);

	  //Shoot right ray
	  float2 d_right = (make_float2(right_pixel))/make_float2(screen) * 2.f - 1.f;
	  float3 ray_dir_right = normalize(d_right.x*U + d_right.y*V + W);

	  optix::Ray ray_right = optix::make_Ray(ray_origin, ray_dir_right, radiance_ray_type, scene_epsilon, RT_DEFAULT_MAX);
	  PerRayData_radiance prd_right;
	  prd_right.importance = 1.f;
	  prd_right.depth = 0;

	  //rtTrace(top_object, ray_left, prd_right); 
	  //float lum_right = luminance(prd_right.result); 

	  //Shoot top ray
	  float2 d_top = (make_float2(top_pixel))/make_float2(screen) * 2.f - 1.f;
	  float3 ray_dir_top = normalize(d_top.x*U + d_top.y*V + W);

	  optix::Ray ray_top = optix::make_Ray(ray_origin, ray_dir_top, radiance_ray_type, scene_epsilon, RT_DEFAULT_MAX);
	  PerRayData_radiance prd_top;
	  prd_top.importance = 1.f;
	  prd_top.depth = 0;

	  //rtTrace(top_object, ray_left, prd_top); 
	  //float lum_top = luminance(prd_top.result);

	  //Shoot bottom ray
	  float2 d_bottom = (make_float2(bottom_pixel))/make_float2(screen) * 2.f - 1.f;
	  float3 ray_dir_bottom = normalize(d_bottom.x*U + d_bottom.y*V + W);

	  optix::Ray ray_bottom = optix::make_Ray(ray_origin, ray_dir_bottom, radiance_ray_type, scene_epsilon, RT_DEFAULT_MAX);
	  PerRayData_radiance prd_bottom;
	  prd_bottom.importance = 1.f;
	  prd_bottom.depth = 0;

	  //rtTrace(top_object, ray_left, prd_bottom); 
	  //float lum_bottom = luminance(prd_bottom.result); 

	  //Shoot delayed ray at centre pixel
	  PerRayData_radiance prd_delayed;
	  prd_delayed.importance = 1.f;
	  prd_delayed.depth = 0;
	  
	  //Shoot all rays together
	  
	  rtTrace(top_object, ray, prd);
	  rtTrace(top_object, ray_left, prd_left);
	  rtTrace(top_object, ray_right, prd_right);
	  rtTrace(top_object, ray_top, prd_top);
	  rtTrace(top_object, ray_bottom, prd_bottom);

	  float lum_centre = luminance(prd.result);
	  float lum_left = luminance(prd_left.result);
	  float lum_right = luminance(prd_right.result);
	  float lum_top = luminance(prd_top.result);
	  float lum_bottom = luminance(prd_bottom.result);

	  uchar4 old_color_char = deep_buffer[rand_pixel].return_sample_at_age(0);
	  float3 old_color = make_float3(old_color_char.z, old_color_char.y, old_color_char.x)*make_float3(1.0f/255.99f);
	  float lum_temporal = luminance(old_color); 
	  
	  //Add results to buffers
	  
	  if(busy_buffer[rand_pixel] == 0) //If pixel is free
	  {
		atomicExch(&busy_buffer[rand_pixel], 1); //Set pixel to busy
		output_buffer[rand_pixel] = make_color(prd.result);
	    color_buffer[rand_pixel] = make_color(prd.result);
	    G_buffer[rand_pixel] = make_float3(prd.hit_point.x, prd.hit_point.y, prd.hit_point.z);
	    stencil_buffer[rand_pixel] = 1;
		atomicExch(&busy_buffer[rand_pixel], 0); //Set pixel to free. 

		//Calculate time elapsed
		clock_t time_stamp = deep_buffer[rand_pixel].get_sample_time();
		clock_t time_elapsed = clock() - time_stamp;
		//rtPrintf("%d clicks \t %f seconds since last sample \n", time_elapsed, ((float)time_elapsed/CLOCKS_PER_SEC)*time_view_scale);
		deep_buffer[rand_pixel].set_sample_time();

		//Calculate appropriate gradients
		float g_x = (abs(lum_left - lum_centre) + abs(lum_centre - lum_right))/2 ;
		float g_y = (abs(lum_bottom - lum_centre) + abs(lum_centre - lum_top))/2 ;
		float g_t = (abs(lum_temporal - lum_centre)) / abs((float)time_elapsed/CLOCKS_PER_SEC);
		g_t /= time_view_scale;

		//Add sample to deep buffer
		//deep_buffer[rand_pixel].add_sample_simple(make_color(prd.result));
		
		//Add crosshair to deep buffer  
		deep_buffer[rand_pixel].add_sample(make_color(prd.result), g_x, g_y, g_t); 
		//deep_buffer[rand_pixel].set_sample_time();

		float3 gradient = deep_buffer[rand_pixel].get_gradients();

		float3 extent;
		extent.x = extent.y = extent.z = 0.0f;
		float vs = 0.04;
		float rl = 100.0/ ((float(tile_size) * (float)tile_size) * (float)0.07); 
		vs = 1/rl; 
		//rtPrintf("%f \n", vs); 
		if(gradient.x == 0.0f || gradient.y == 0.0f || gradient.z == 0.0f)
		{
			extent.x = extent.y = extent.z = 0.0f;
		}
		else
		{
			extent.x = pow((gradient.y * gradient.z * vs)/(gradient.x * gradient.x),0.33f);
			extent.y = pow((gradient.x * gradient.z * vs)/(gradient.y * gradient.y),0.33f);
			extent.z = pow((gradient.x * gradient.y * vs)/(gradient.z * gradient.z),0.33f);
		}

		//rtPrintf("extent = %f %f %f \n", extent.x, extent.y, extent.z); 
		deep_buffer[rand_pixel].set_extents(extent); 
		
	 
	  }

	  
  }
}

static __device__ __inline__ void shoot_rays()
{
	size_t2 screen = output_buffer.size();

	volatile unsigned int seed  = rnd_seeds[ launch_index ]; // volatile workaround for cuda 2.0 bug
  unsigned int new_seed  = seed;
  float uu = rnd( new_seed );
  float vv = rnd(new_seed);
  uint2 random_index = make_uint2(uu*screen.x, vv*screen.y);

  rnd_seeds[ launch_index ] = new_seed;

  //Shoot ray offsetting from centre pixel of leaf tiles
  //int rnd_tile_index = rnd(new_seed) * 256; 

  //rtPrintf(" Tile = %d \n", rnd_tile_index);

  //Get centre pixel associated with thread
  uint2 centre_pixel = make_uint2(leaf_tile_indices[launch_index.x].x, 
								  leaf_tile_indices[launch_index.x].y);
  
  //rtPrintf("Tile = %u \n", launch_index.x);

  //Launch X rays per thread
  for(int i = 0;i<100;i++)
  {
	  float x_offset = rnd(new_seed);
	  float y_offset = rnd(new_seed);
	  float x_direction = rnd(new_seed);
	  float y_direction = rnd(new_seed);

	  unsigned int tile_size = leaf_tile_sizes[launch_index.x];

	  uint2 start_pixel;
	  start_pixel.x = centre_pixel.x - tile_size /2u;
	  start_pixel.y = centre_pixel.y - tile_size /2u;

	  x_offset *= tile_size;
	  y_offset *= tile_size;
	  uint2 total_offset = make_uint2(x_offset, y_offset);
	  uint2 rand_pixel = start_pixel + total_offset;

	  //Now actually shoot the ray
	  float2 d = (make_float2(rand_pixel))/make_float2(screen) * 2.f - 1.f;
	  float3 ray_origin = eye;
	  float3 ray_direction = normalize(d.x*U + d.y*V + W);

	  optix::Ray ray = optix::make_Ray(ray_origin, ray_direction, radiance_ray_type, scene_epsilon, RT_DEFAULT_MAX);

	  PerRayData_radiance prd;
	  prd.importance = 1.f;
	  prd.depth = 0;

	  rtTrace(top_object, ray, prd);

	  if(busy_buffer[rand_pixel] == 0) //If pixel is free
	  {
		atomicExch(&busy_buffer[rand_pixel], 1); //Set pixel to busy
		output_buffer[rand_pixel] = make_color(prd.result);
	    color_buffer[rand_pixel] = make_color(prd.result);
	    G_buffer[rand_pixel] = make_float3(prd.hit_point.x, prd.hit_point.y, prd.hit_point.z);
	    stencil_buffer[rand_pixel] = 1;
		atomicExch(&busy_buffer[rand_pixel], 0); //Set pixel to free. 
	  }

	  
  }

}

static __device__ __inline__ void reproject()
{
	volatile unsigned int seed  = rnd_seeds[ launch_index ]; // volatile workaround for cuda 2.0 bug
  unsigned int new_seed  = seed;
  uint2 centre_pixel = make_uint2(leaf_tile_indices[launch_index.x].y, 
								  leaf_tile_indices[launch_index.x].x); //Switch .x and .y 
  //Reprojection

	  glm::mat4 Projection = glm::perspective(60.0f, 1.0f, 0.01f, 100.0f);
	  glm::mat4 T = glm::translate(glm::mat4(1.f),glm::vec3(-eye.x, -eye.y, -eye.z));

	  glm::vec3 eye_position = glm::vec3(eye.x, eye.y, eye.z);
	  glm::vec3 D = glm::vec3(W.x, W.y, W.z);
	  glm::vec3 centre = eye_position + D;
	  glm::vec3 up = glm::normalize(glm::vec3(0.0,1.0,0.0));
	  glm::mat4 View = glm::lookAt(eye_position, centre, up);
	  //glm::mat4 ViewProjection = Projection * View;
	  glm::mat4 ViewProjection = View*T;

  for(int i=0; i<400;i++)
  {
	  float x_offset = rnd(new_seed);
	  float y_offset = rnd(new_seed);
	  float x_direction = rnd(new_seed);
	  float y_direction = rnd(new_seed);

	  //uint2 rnd_pixel_within_tile = centre_pixel;
	  uint2 total_offset;

	  x_offset *= leaf_tile_sizes[launch_index.x]; //Scale the offset by the size of the tile
	  y_offset *= leaf_tile_sizes[launch_index.x];

	  if(x_direction > 0.5 && y_direction > 0.5)//Offset in +x direction
	  {
		  total_offset = make_uint2(x_offset, y_offset);
	  }
	  else if(x_direction > 0.5 && y_direction <= 0.5)
	  {
		  total_offset = make_uint2(x_offset, y_offset *-1);
	  }
	  else if(x_direction <= 0.5 && y_direction > 0.5)
	  {
		  total_offset = make_uint2(x_offset * -1, y_offset);
	  }
	  else if(x_direction <= 0.5 && y_direction <= 0.5)
	  {
		  total_offset = make_uint2(x_offset* -1, y_offset* -1);
	  }


	  //Rand pixel is random pixel within tile
	  uint2 rand_pixel = centre_pixel + total_offset;
	  glm::vec4 old_position; /*= glm::vec4(G_buffer[rand_pixel].x, 
										 G_buffer[rand_pixel].y, 
										 G_buffer[rand_pixel].z,
										 1.0f);*/

	  glm::vec4 new_position;
	  
	  /*
	  int curr;
	  do{
		  curr = atomicCAS(&busy_buffer[rand_pixel], 0, 1);
	  }while(curr != 1) //loop until the pixel is locked. 
	  */
	  if(busy_buffer[rand_pixel] == 0) //If pixel is free
	  {
		  atomicExch(&busy_buffer[rand_pixel], 1); //Set pixel as busy
		  old_position = glm::vec4(G_buffer[rand_pixel].x,
								   G_buffer[rand_pixel].y,
								   G_buffer[rand_pixel].z,
								   1.0f);
		  //Set the pixel as free after reprojection or right now?
		  atomicExch(&busy_buffer[rand_pixel], 0); //Set pixel as free
	  }
	  

	  
	  if(stencil_buffer[rand_pixel] == 1 || busy_buffer[rand_pixel] == 1) //If value is new. Do not reproject
	  {
		  new_position = old_position;
	  }
	  else //If value is old, reproject
	  {
		  new_position = ViewProjection * old_position;
		  stencil_buffer[rand_pixel] = 2;
		  
		  uint2 new_pixel;//stores the new pixel value

		  if(old_position.x == 100.0f &&
			 old_position.y == 100.0f &&
			 old_position.z == 100.0f) //If the old G buffer stores a miss
		  {
			  new_position = old_position;
			  new_pixel = rand_pixel;
		  }
		  else
		  {
			  new_position = ViewProjection * old_position;
			  float2 new_pixel_float;

			  new_pixel_float.x = ((new_position.x+1.0)/2.0)*512.0;
			  new_pixel_float.y = ((new_position.y+1.0)/2.0)*512.0;

			  //Static cast to uint as loss of precision may occur
			  new_pixel.x = static_cast<unsigned int>(new_pixel_float.x);
			  new_pixel.y = static_cast<unsigned int>(new_pixel_float.y);
		  }

		  //Copy values from old pixel to new pixel
		  if(busy_buffer[new_pixel] == 0 && busy_buffer[rand_pixel] == 0) //If pixel is free
		  {
			  atomicExch(&busy_buffer[new_pixel], 1); //Set new pixel to busy
			  atomicExch(&busy_buffer[rand_pixel], 1); //Set old pixel to busy
			  color_buffer[new_pixel] = color_buffer[rand_pixel];
			  output_buffer[new_pixel] = output_buffer[rand_pixel];
			  G_buffer[new_pixel] = G_buffer[rand_pixel];
			  atomicExch(&busy_buffer[new_pixel], 0); //Set new pixel to free
			  atomicExch(&busy_buffer[rand_pixel], 0); //Free the old pixel too.

		  }

	  }

  }
}

static __device__ __inline__ void calculate_parent_variance()
{
	uint2 centre_pixel = make_uint2(parent_tile_indices[launch_index.x].y, 
								  parent_tile_indices[launch_index.x].x);
	unsigned int tile_size = parent_tile_sizes[launch_index.x];

	uint2 start_pixel;
	start_pixel.x = centre_pixel.x - tile_size/2u;
	start_pixel.y = centre_pixel.y - tile_size/2u;

	uint2 end_pixel;
	end_pixel.x = centre_pixel.x + tile_size/2u;
	end_pixel.y = centre_pixel.y + tile_size/2u; 

	uint2 index_pixel = start_pixel;

	uchar4 mean_color;
	uchar4 char_color;
	float3 color;
	float n = 0.0;

	float3 mean; mean.x = mean.y = mean.z = 0.0;
	float3 m2; m2.x = m2.y = m2.z = 0.0;
	float3 variance; 
	for(unsigned int i=0; i<tile_size; i++)
	{
		for(unsigned int j=0; j<tile_size; j++)
		{
			n++;
			uint2 offset = make_uint2(i,j);
			uint2 index = start_pixel + offset;
			
			//char_color = color_buffer[index];
			//char_color = deep_buffer[index].return_average_color();
			char_color = deep_buffer[index].weighted_average_color();
			color = make_float3(char_color.z, char_color.y, char_color.x)*make_float3(1.0f/255.99f);
			
			//rtPrintf("color = %f \t %f \t %f \n", color.x, color.y, color.z);

			float3 delta = color - mean;
			mean += delta/n;
			m2 += delta*(color - mean);
			//mean = prev_mean + (color - prev_mean)/n;

		}
	}

	variance = m2/(n-1);
	float v = luminance(variance);

	parent_variance_buffer[launch_index.x] = v; //Write the variance value into the variance buffer 

	//rtPrintf("v = \t%f \t \t %u \t %u \n", v, launch_index.x, launch_index.y);

	//rtPrintf("m2 = %f \t %f \t %f \n", m2.x, m2.y, m2.z);
	//rtPrintf("diff = %f \t %f \t %f \n", diff.x, diff.y, diff.z);
	//rtPrintf("sq sum = %f %f %f \n", squared_sum.x*n, squared_sum.y*n, squared_sum.z*n);
	//rtPrintf("V = %f %f %f \n", variance.x, variance.y, variance.z);
}

static __device__ __inline__ void calculate_variance(bool leaf)
{
	uint2 centre_pixel;
	unsigned int tile_size;
	if(leaf)
	{
		centre_pixel = make_uint2(leaf_tile_indices[launch_index.x].y,
									leaf_tile_indices[launch_index.x].x);
		tile_size = leaf_tile_sizes[launch_index.x];
	}
	else
	{
		centre_pixel = make_uint2(parent_tile_indices[launch_index.x].y,
									parent_tile_indices[launch_index.x].x);
		tile_size = parent_tile_sizes[launch_index.x];
	}
	/*
	uint2 centre_pixel = make_uint2(leaf_tile_indices[launch_index.x].y, 
								  leaf_tile_indices[launch_index.x].x);
	unsigned int tile_size = leaf_tile_sizes[launch_index.x];
	*/
	uint2 start_pixel;
	start_pixel.x = centre_pixel.x - tile_size/2u;
	start_pixel.y = centre_pixel.y - tile_size/2u;

	uint2 end_pixel;
	end_pixel.x = centre_pixel.x + tile_size/2u;
	end_pixel.y = centre_pixel.y + tile_size/2u; 

	uint2 index_pixel = start_pixel;

	uchar4 mean_color;
	uchar4 char_color;
	float3 color;
	float n = 0.0;

	float3 mean; mean.x = mean.y = mean.z = 0.0;
	float3 m2; m2.x = m2.y = m2.z = 0.0;
	float3 variance; 
	for(unsigned int i=0; i<tile_size; i++)
	{
		for(unsigned int j=0; j<tile_size; j++)
		{
			n++;
			uint2 offset = make_uint2(i,j);
			uint2 index = start_pixel + offset;
			
			//char_color = color_buffer[index];
			//char_color = deep_buffer[index].return_average_color();
			char_color = deep_buffer[index].weighted_average_color();
			color = make_float3(char_color.z, char_color.y, char_color.x)*make_float3(1.0f/255.99f);
			
			float3 delta = color - mean;
			mean += delta/n;
			m2 += delta*(color - mean);
			//mean = prev_mean + (color - prev_mean)/n;

		}
	}

	variance = m2/(n-1);
	float v = luminance(variance);

	if(leaf)
	{
		variance_buffer[launch_index.x] = v;
	}
	else
	{
		parent_variance_buffer[launch_index.x] = v; 
	}

	//variance_buffer[launch_index.x] = v; //Write the variance value into the variance buffer 

}

static __device__ __inline__ void calculate_gradients()
{
	
	uint2 centre_pixel = make_uint2(leaf_tile_indices[launch_index.x].y, 
								  leaf_tile_indices[launch_index.x].x);
	unsigned int tile_size = leaf_tile_sizes[launch_index.x];
	
	uint2 start_pixel;
	start_pixel.x = centre_pixel.x - tile_size/2u;
	start_pixel.y = centre_pixel.y - tile_size/2u;

	float n = 0.0;

	float3 gradient;
	gradient.x = gradient.y = gradient.z = 0.0f;

	for(unsigned int i=0; i<tile_size; i++)
	{
		for(unsigned int j=0; j<tile_size; j++)
		{
			n++;
			uint2 offset = make_uint2(i,j);
			uint2 index = start_pixel + offset;
			
			//char_color = color_buffer[index];
			//char_color = deep_buffer[index].return_average_color();
			//char_color = deep_buffer[index].weighted_average_color();
			//color = make_float3(char_color.z, char_color.y, char_color.x)*make_float3(1.0f/255.99f);
			
			gradient += deep_buffer[index].get_gradients();

		}
	}

	gradient *= make_float3(1.0/n); 

	gradient_buffer[launch_index.x] = gradient; 

	//rtPrintf("avg. gradient = %f %f %f \t launch_index = %u \n", gradient.x, gradient.y, gradient.z, launch_index.x); 

}

static __device__ __inline__ void reconstruction()
{
	uint2 centre_pixel = make_uint2(leaf_tile_indices[launch_index.x].y, 
								  leaf_tile_indices[launch_index.x].x);
	unsigned int tile_size = leaf_tile_sizes[launch_index.x];
	
	uint2 start_pixel;
	start_pixel.x = centre_pixel.x - tile_size/2u;
	start_pixel.y = centre_pixel.y - tile_size/2u;

	//Define all Filter variables
	float sig = 0.03f;
	

	for(unsigned int i=0; i<tile_size; i++)
	{
		for(unsigned int j=0; j<tile_size; j++)
		{
			
			uint2 offset = make_uint2(i,j);
			uint2 index = start_pixel + offset;
			
			float3 extents = deep_buffer[index].get_extents();
			//rtPrintf("%d %d %d \n", (int)extents.x, (int)extents.y, (int)extents.z); 
			uint2 filter_size = make_uint2((unsigned int)extents.x, (unsigned int)extents.y);

			uint2 first_pixel;
			first_pixel.x = index.x - filter_size.x/2u;
			first_pixel.y = index.y - filter_size.y/2u;

			//Loop through all the pixels in the neighborhood
			float total_weight = 0.0f;
			uchar4 total_color; 
			float3 tot_color;
			tot_color.x = tot_color.y = tot_color.z = 0.0f; 
			//unsigned int min = filter_size.x > 0u ? 6u : filter_size.x;
			if(filter_size.x == 0u || filter_size.y == 0u)
			{
				//rtPrintf("!! \n"); 
				output_buffer[index] = color_buffer[index]; 
				//float3 gradient = deep_buffer[index].get_gradients();
				//rtPrintf("%f %f %f \n", gradient.x, gradient.y, gradient.z); 
			}
			else
			{
			for(unsigned int h = 0; h<8u; h++)
			{
				for(unsigned int v = 0; v<8u; v++)
				{
					float g;
					g = (1.0f)/(2.0f*PI*sig*sig);
					int x = (int)h;
					int y = (int)v;
					float e = exp(((x*x + y*y)/2.0f*sig*sig)* -1.0f);
					g *= e;
					
					rtPrintf("%d %d \n", x, y); 
					//top pixel
					int2 top_pixel = make_int2((int)index.x, (int)index.y+y);
					if(top_pixel.x >= 0 && top_pixel.x <= 512 &&
					   top_pixel.y >= 0 && top_pixel.y <= 512)
					{
						uint2 top_px = make_uint2((unsigned int)top_pixel.x, (unsigned int)top_pixel.y);
						//uchar4 top_color = deep_buffer[top_px].return_sample_at_age(1);
						uchar4 top_color = color_buffer[top_px];
						//uchar4 top_color = deep_buffer[top_px].weighted_average_color();
						float3 top_color_float = make_float3(top_color.z, top_color.y, top_color.x)*make_float3(1.0f/255.99f);
						top_color_float *= make_float3(g);
						tot_color += top_color_float;
						total_weight += g;
					}
					//bottom pixel
					int2 bot_pixel = make_int2((int)index.x, (int)index.y-y);
					if(bot_pixel.x >= 0 && bot_pixel.x <= 512 &&
					   bot_pixel.y >= 0 && bot_pixel.y <= 512)
					{
						uint2 bot_px = make_uint2((unsigned int)bot_pixel.x, (unsigned int)bot_pixel.y);
						//uchar4 bot_color = deep_buffer[bot_px].return_sample_at_age(1);
						uchar4 bot_color = color_buffer[bot_px];
						//uchar4 bot_color = deep_buffer[bot_px].weighted_average_color();
						float3 bot_color_float = make_float3(bot_color.z, bot_color.y, bot_color.x)*make_float3(1.0f/255.99f);
						bot_color_float *= make_float3(g);
						tot_color += bot_color_float;
						total_weight += g;
					}
					//right pixel
					int2 right_pixel = make_int2((int)index.x + x, (int)index.y);
					if(right_pixel.x >= 0 && right_pixel.x <= 512 &&
					   right_pixel.y >= 0 && right_pixel.y <= 512)
					{
						uint2 right_px = make_uint2((unsigned int)right_pixel.x, (unsigned int)right_pixel.y);
						//uchar4 right_color = deep_buffer[right_px].return_sample_at_age(1);
						uchar4 right_color = color_buffer[right_px];
						//uchar4 right_color = deep_buffer[right_px].weighted_average_color();
						float3 right_color_float = make_float3(right_color.z, right_color.y, right_color.x)*make_float3(1.0f/255.99f);
						right_color_float *= make_float3(g);
						tot_color += right_color_float;
						total_weight += g;
					}
					//left pixel
					int2 left_pixel = make_int2((int)index.x - x, (int)index.y);
					if(left_pixel.x >= 0 && left_pixel.x <= 512 &&
					   left_pixel.y >= 0 && left_pixel.y <= 512)
					{
						uint2 left_px = make_uint2((unsigned int)left_pixel.x, (unsigned int)left_pixel.y);
						//uchar4 left_color = deep_buffer[left_px].return_sample_at_age(1);
						uchar4 left_color = color_buffer[left_px];
						//uchar4 left_color = deep_buffer[left_px].weighted_average_color();
						float3 left_color_float = make_float3(left_color.z, left_color.y, left_color.x)*make_float3(1.0f/255.99f);
						left_color_float *= make_float3(g);
						tot_color += left_color_float;
						total_weight += g;
					}
					
					
				}
			}//for loop ends here
			//Finally, add the weighted average colors to the buffer
			tot_color *= make_float3(1.0f/total_weight);
			uchar4 final_color = make_color(tot_color);
			output_buffer[index] = final_color;
			//color_buffer[index] = final_color; 
			}//else ends here
			

		}
	}
}

static __device__ __inline__ void gather_x_afr()
{
	uint2 index = make_uint2(launch_index.x, launch_index.y);

	float total_weight = 0.0f;
	float3 total_color;
	total_color.x = total_color.y = total_color.z = 0.0f;

	//l to r sweep 
	for(int i= -3 ; i<4; i++)
	{
		int2 int_index = make_int2((int)index.x + i, (int)index.y); 
		uint2 new_index = make_uint2((unsigned int)int_index.x, (unsigned int)int_index.y);
		//rtPrintf("%u %u \n", new_index.x, new_index.y); 
		if(new_index.x < 512u && new_index.x >= 0u) //Within bounds
		{
			uchar4 temp_color_char = deep_buffer[new_index].return_sample_at_age(0);
			//uchar4 temp_color_char = color_buffer[new_index]; 
			float3 temp_color_float = make_float3(temp_color_char.z, temp_color_char.y, temp_color_char.x) * make_float3(1.0f/255.99f);
			temp_color_float *= make_float3(gaussian_7[i+3]);
			total_weight += gaussian_7[i+3];
			total_color += temp_color_float;  

			float3 extents = deep_buffer[new_index].get_extents();
		}
	}

	//output_buffer[index] = make_color(total_color); 
	total_color *= make_float3(1.0f/total_weight);
	gaussian_x_buffer[index] = make_color(total_color); 
}

static __device__ __inline__ void gather_y_afr()
{
	uint2 index = make_uint2(launch_index.x, launch_index.y);

	float total_weight = 0.0f;
	float3 total_color;
	total_color.x = total_color.y = total_color.z = 0.0f;

	//bottom to top sweep 
	for(int i= -3 ; i<4; i++)
	{
		uint2 new_index = make_uint2(index.x, index.y + (unsigned int)i);
		
		if(new_index.y < 512u && new_index.y >= 0u) //Within bounds
		{
			//uchar4 temp_color_char = deep_buffer[new_index].return_sample_at_age(0);
			uchar4 temp_color_char = gaussian_x_buffer[new_index]; 
			float3 temp_color_float = make_float3(temp_color_char.z, temp_color_char.y, temp_color_char.x) * make_float3(1.0f/255.99f);
			temp_color_float *= make_float3(gaussian_7[i+3]);
			total_weight += gaussian_7[i+3];
			total_color += temp_color_float;  
		}
	}

	//output_buffer[index] = make_color(total_color); 
	total_color *= make_float3(1.0f/total_weight);
	gaussian_y_buffer[index] = make_color(total_color); 
}

static __device__ __inline__ void gather_t_afr()
{
	uint2 index = make_uint2(launch_index.x, launch_index.y);
	float total_weight = 0.0f;
	float3 total_color;
	total_color.x = total_color.y = total_color.z = 0.0f;

	total_color += make_float3(gaussian_y_buffer[index].z, gaussian_y_buffer[index].y, gaussian_y_buffer[index].x) * make_float3(1.0f/255.99f);


	for(int i=0; i<2; i++)
	{
		uchar4 temp_color_char = deep_buffer[index].return_sample_at_age(i);
		float3 temp_color_float = make_float3(temp_color_char.z, temp_color_char.y, temp_color_char.x) * make_float3(1.0f/255.99f);
		temp_color_float *= make_float3(gaussian_3[i+1]);
		total_weight += gaussian_3[i+1];
		total_color += temp_color_float;
	}
	total_color *= make_float3(1.0f/total_weight);
	output_buffer[index] = make_color(total_color); 
}


static __device__ __inline__ void gather()
{

	uint2 index = make_uint2(launch_index.x, launch_index.y);

	float sig = 0.3f;

	float3 extents = deep_buffer[index].get_extents();
	uint2 filter_size = make_uint2((unsigned int)extents.x, (unsigned int)extents.y);

	float total_weight = 0.0f;
	uchar4 total_color; 
	float3 tot_color;
	tot_color.x = tot_color.y = tot_color.z = 0.0f; 
	//unsigned int min = filter_size.x > 0u ? 6u : filter_size.x;
	if(filter_size.x == 0u || filter_size.y == 0u)
	{
		output_buffer[index] = color_buffer[index]; 
		//output_buffer[index] = make_color(make_float3(0.0f, 1.0f, 0.0f));
		//float3 gradient = deep_buffer[index].get_gradients();
		//rtPrintf("%f %f %f \n", gradient.x, gradient.y, gradient.z); 
	}
	else
	{
		//rtPrintf("filter_size : %u  %u  \n", filter_size.x, filter_size.y); 
		for(unsigned int i = 0u; i< 5u; i++)
		{
			for(unsigned int j = 0u; j< 5u; j++)
			{
				float g;
				g = (1.0f)/(2.0f*PI*sig*sig);
				int x = (int)i;
				int y = (int)j;
				float e = exp(((x*x + y*y)/2.0f*sig*sig)* -1.0f);
				g *= e;
			    //top pixel
				int2 top_pixel = make_int2((int)index.x, (int)index.y+y);
				if(top_pixel.x >= 0 && top_pixel.x <= 512 &&
				   top_pixel.y >= 0 && top_pixel.y <= 512)
				{
					uint2 top_px = make_uint2((unsigned int)top_pixel.x, (unsigned int)top_pixel.y);
					//uchar4 top_color = deep_buffer[top_px].return_sample_at_age(1);
					uchar4 top_color = color_buffer[top_px];
					//uchar4 top_color = deep_buffer[top_px].weighted_average_color();
					float3 top_color_float = make_float3(top_color.z, top_color.y, top_color.x)*make_float3(1.0f/255.99f);
					top_color_float *= make_float3(g);
					tot_color += top_color_float;
					total_weight += g;
				}
				//bottom pixel
				int2 bot_pixel = make_int2((int)index.x, (int)index.y-y);
				if(bot_pixel.x >= 0 && bot_pixel.x <= 512 &&
				   bot_pixel.y >= 0 && bot_pixel.y <= 512)
				{
					uint2 bot_px = make_uint2((unsigned int)bot_pixel.x, (unsigned int)bot_pixel.y);
					//uchar4 bot_color = deep_buffer[bot_px].return_sample_at_age(1);
					uchar4 bot_color = color_buffer[bot_px];
					//uchar4 bot_color = deep_buffer[bot_px].weighted_average_color();
					float3 bot_color_float = make_float3(bot_color.z, bot_color.y, bot_color.x)*make_float3(1.0f/255.99f);
					bot_color_float *= make_float3(g);
					tot_color += bot_color_float;
					total_weight += g;
				}
				//right pixel
				int2 right_pixel = make_int2((int)index.x + x, (int)index.y);
				if(right_pixel.x >= 0 && right_pixel.x <= 512 &&
				   right_pixel.y >= 0 && right_pixel.y <= 512)
				{
					uint2 right_px = make_uint2((unsigned int)right_pixel.x, (unsigned int)right_pixel.y);
					//uchar4 right_color = deep_buffer[right_px].return_sample_at_age(1);
					uchar4 right_color = color_buffer[right_px];
					//uchar4 right_color = deep_buffer[right_px].weighted_average_color();
					float3 right_color_float = make_float3(right_color.z, right_color.y, right_color.x)*make_float3(1.0f/255.99f);
					right_color_float *= make_float3(g);
					tot_color += right_color_float;
					total_weight += g;
				}
				//left pixel
				int2 left_pixel = make_int2((int)index.x - x, (int)index.y);
				if(left_pixel.x >= 0 && left_pixel.x <= 512 &&
				   left_pixel.y >= 0 && left_pixel.y <= 512)
				{
					uint2 left_px = make_uint2((unsigned int)left_pixel.x, (unsigned int)left_pixel.y);
					//uchar4 left_color = deep_buffer[left_px].return_sample_at_age(1);
					uchar4 left_color = color_buffer[left_px];
					//uchar4 left_color = deep_buffer[left_px].weighted_average_color();
					float3 left_color_float = make_float3(left_color.z, left_color.y, left_color.x)*make_float3(1.0f/255.99f);
					left_color_float *= make_float3(g);
					tot_color += left_color_float;
					total_weight += g;
				}

				
			}
		}//for loop ends
		tot_color *= make_float3(1.0f/total_weight);
		uchar4 final_color = make_color(tot_color);
		output_buffer[index] = final_color;
	}// else ends 

}

static __device__ __inline__ void scatter()
{
	uint2 index = make_uint2(launch_index.x, launch_index.y);

	if(stencil_buffer[index] == 1) //Scatter only if the value is new
	{
		float sig = 0.03f;

		float3 extents = deep_buffer[index].get_extents();
		uint2 filter_size = make_uint2((unsigned int)extents.x, (unsigned int)extents.y);

		//color to be scattered in float3 
		float3 scatter_color = make_float3(color_buffer[index].z, color_buffer[index].y, color_buffer[index].x) * make_float3(1.0f/255.99f); 

		for(unsigned int i=0u; i<5u; i++)
		{
			for(unsigned int j=0u; j<5u; j++)
			{
				float g;
				g = (1.0f)/(2.0f*PI*sig*sig);
				int x = (int)i;
				int y = (int)j;
				float e = exp(((x*x + y*y)/2.0f*sig*sig)* -1.0f);
				g *= e;

				//top pixel
				uint2 top_px = make_uint2(index.x, (index.y + (unsigned int)y));
				if(top_px.x >= 0u && top_px.x <= 512u &&
				   top_px.y >= 0u && top_px.y <= 512u)
				{
					float3 weighted_color = scatter_color * make_float3(g); 
					//wixi_buffer[top_px] += weighted_color;
					//wi_buffer[top_px] += g;
					//atomicAdd(&wi_buffer[top_px], g);
					//rtPrintf("wi_buffer \t  %f \n", wi_buffer[top_px]); 
				}
				
				//bottom pixel
				uint2 bot_px = make_uint2(index.x, (index.y - (unsigned int)y));
				if(bot_px.x >= 0u && bot_px.x <= 512u &&
				   bot_px.y >= 0u && bot_px.y <= 512u)
				{
					float3 weighted_color = scatter_color * make_float3(g);
					//wixi_buffer[bot_px] += weighted_color;
					//wi_buffer[bot_px] +=g;
				}

				//right pixel
				

			}
		}//outer for loop ends 
	}//if statement ends
}


RT_PROGRAM void frameless_pinhole_camera()
{
	size_t2 screen = output_buffer.size();
	volatile unsigned int seed  = rnd_seeds[ launch_index ];
  unsigned int new_seed  = seed;
  float uu = rnd( new_seed );
  float vv = rnd(new_seed);
  uint2 random_index = make_uint2(uu*screen.x, vv*screen.y);

  rnd_seeds[ launch_index ] = new_seed;

	float3 result = trace(make_float2(random_index));
	output_buffer[random_index] = make_color(result);
	color_buffer[random_index] = make_color(result);
}

RT_PROGRAM void pinhole_camera()
{
	shoot_afr();
	//shoot_new();
	//calculate_variance(true);
	//calculate_gradients();
	/*
	if(launch_index.x <= (unsigned int)number_of_parent_tiles)
	{
		calculate_variance(false);
	}
	*/
	
	
}

RT_PROGRAM void new_pinhole_camera()
{
	if(launch_index.y == 0u)
	{
		calculate_variance(true);
		calculate_gradients();
	}
	else if(launch_index.y == 1u && launch_index.x <= (unsigned int)number_of_parent_tiles)
	{
		calculate_variance(false);
	}
	else
	{
		shoot_new();
	}
}

RT_PROGRAM void a_pinhole_camera()
{
 #ifdef TIME_VIEW
  clock_t t0 = clock(); 
 #endif

  //shoot();
  //shoot_new();
  
  //shoot_rays();
  //reproject();

  
  
  if(launch_index.y >= 3u)
  {
	  clock_t time_my = clock();
	  shoot_new();
	  time_my = clock() - time_my;
	  //rtPrintf(" Shoot = %f \n", (float)time_my/CLOCKS_PER_SEC * time_view_scale);
  }
  else if(launch_index.y == 0u)
  {
	  clock_t time_my = clock();
	  calculate_variance(true); 
	  //rtPrintf("Leaf! %u \n", launch_index.x);
	  time_my = clock() - time_my;
	  //rtPrintf(" Leaf = %f \n", (float)time_my/CLOCKS_PER_SEC * time_view_scale);
  }
  else if(launch_index.x <= (unsigned int)number_of_parent_tiles &&
	  launch_index.y == 1u)
  {
	  clock_t time_my = clock();
	  //calculate_parent_variance();
	  calculate_variance(false);
	  //rtPrintf("Parent! \n");
	  time_my = clock() - time_my;
	  //rtPrintf(" Parent = %f \n", (float)time_my/CLOCKS_PER_SEC * time_view_scale);
  }
  else if(launch_index.y == 2u)
  {
	  clock_t time_my = clock();
	  calculate_gradients();
	  //rtPrintf("Gradient! %u \n", launch_index.x);
	  time_my = clock() - time_my;
	  //rtPrintf(" Gradient = %f \n", (float)time_my/CLOCKS_PER_SEC * time_view_scale);
  }
  //gradient sampling & calculations
  

#ifdef TIME_VIEW
  clock_t t1 = clock(); 
 
  float expected_fps   = 1.0f;
  float pixel_time     = ( t1 - t0 ) * time_view_scale * expected_fps;
  output_buffer[launch_index] = make_color( make_float3(  pixel_time ) ); 
#else
  /*
  //temp_buffer stores previous pixel's color
  float3 difference_color = prd.result - temp_buffer[random_index];
  temp_buffer[random_index] = prd.result;

  //output_buffer[random_index] = make_color(difference_color);
  output_buffer[random_index] = make_color(prd.result);
  color_buffer[random_index] = make_color(prd.result);
  */

#endif

}



RT_PROGRAM void exception()
{
  const unsigned int code = rtGetExceptionCode();
  rtPrintf( "Caught exception 0x%X at launch index (%d,%d)\n", code, launch_index.x, launch_index.y );
  output_buffer[launch_index] = make_color( bad_color );
}

RT_PROGRAM void parent()
{
	calculate_variance(false);
}

RT_PROGRAM void leaf()
{
	//reproject();
	calculate_variance(true); 
	//calculate_gradients();

	/*
	if(launch_index.y == 1u)
	{
		calculate_variance(true);
	}
	else
	{
		calculate_gradients();
	}
	*/
	
}

RT_PROGRAM void gradient()
{
	calculate_gradients();
}

RT_PROGRAM void reconstruct()
{
	//reconstruction();
	gather(); //Performs gather reconstruction as 512 x 512 pxs
	//gather_x_afr();
	//gather_y_afr();
	//gather_t_afr();
	//scatter();
}