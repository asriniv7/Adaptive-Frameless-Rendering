
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

//-------------------------------------------------------------------------------
//
//  whitted.cpp -- whitted's original sphere scene 
//
//-------------------------------------------------------------------------------

#include <optixu/optixpp_namespace.h>
#include <optixu/optixu_math_namespace.h>
#include "random.h"
#include <iostream>
#include <GLUTDisplay.h>
#include "commonStructs.h"
#include <cstdlib>
#include <cstring>

#include <time.h>
#include <queue>


//#include<glm\glm\glm.hpp>
//#include<glm\glm\vec4.hpp>
//#include<glm\glm\mat4x4.hpp>
//#include<glm\glm\gtc\matrix_transform.hpp>

#include "Tile.h"

using namespace optix;

//-----------------------------------------------------------------------------
// 
// Whitted Scene
//
//-----------------------------------------------------------------------------

class WhittedScene : public SampleScene
{
public:
  WhittedScene() : SampleScene(), m_frame_number( 0 ), m_adaptive_aa( false ), m_width(512u), m_height(512u) {}

  // From SampleScene
  void   initScene( InitialCameraData& camera_data );
  void   trace( const RayGenCameraData& camera_data );
  void   doResize( unsigned int width, unsigned int height );
  void   setDimensions( const unsigned int w, const unsigned int h ) { m_width = w; m_height = h; }
  Buffer getOutputBuffer();
  bool keyPressed(unsigned char key, int x, int y);

  void setAdaptiveAA( bool adaptive_aa ) { m_adaptive_aa = adaptive_aa; }
  void doRetiling();
  void clearStencil(); // calls the clear_stencil method of all leaf tiles
  void retiling(); //Newer version of doRetiling
  void drawTiles(); //draws tiles with gradients 
  void map(); //map buffers before retiling
  void unmap(); //unmap buffers after retiling 
  void retiling_new(int ); //new retiling procedure 

private:
  int getEntryPoint() { return m_adaptive_aa ? AdaptivePinhole: Pinhole; }
  void genRndSeeds(unsigned int width, unsigned int height);

  enum {
    Pinhole = 0,
    AdaptivePinhole = 1,
	Reconstruction = 2,
	Parent = 3,
	Leaf = 4,
	Gradient = 5
  };

  void createGeometry();
  void createSimpleGeometry(); //Create a simple scene - just one edge. 
  void createComplexGeometry(); //Create a complicated scene. 

  void updateGeometry(); //animate the geometry

  Buffer        m_rnd_seeds;
  unsigned int  m_frame_number;
  bool          m_adaptive_aa;

  unsigned int m_width;
  unsigned int m_height;

  Tile root_tile;
  std::vector<Tile*> leaf_tiles;
  std::vector<Tile*> parent_tiles;
  
  //Priority Queues for the tiles
  std::priority_queue<Tile*, std::vector<Tile*>, MoreVariance> leaf_pq; //Max pq for all leaves 
  std::priority_queue<Tile*, std::vector<Tile*>, LessVariance> parent_pq; //Min pq for all aprents 

  int rand_tile_index;

  clock_t t; 
  clock_t cpu_time;
};

void WhittedScene::genRndSeeds( unsigned int width, unsigned int height )
{
  unsigned int* seeds = static_cast<unsigned int*>( m_rnd_seeds->map() );
  fillRandBuffer( seeds, width*height );
  m_rnd_seeds->unmap();
}

void WhittedScene::initScene( InitialCameraData& camera_data )
{
  // context 
  m_context->setRayTypeCount( 2 );
  m_context->setEntryPointCount( 6 ); //Set to 2 initially
  m_context->setStackSize( 4800 );
  m_context->setPrintEnabled(true);
  m_context->setPrintBufferSize(2400);

  m_context["max_depth"]->setInt( 10 );
  m_context["radiance_ray_type"]->setUint( 0 );
  m_context["shadow_ray_type"]->setUint( 1 );
  m_context["frame_number"]->setUint( 0u );
  m_context["scene_epsilon"]->setFloat( 1.e-4f );
  m_context["ambient_light_color"]->setFloat( 0.4f, 0.4f, 0.4f );


  //m_context["output_buffer"]->set( createOutputBuffer(RT_FORMAT_UNSIGNED_BYTE4, m_width, m_height) );
  m_context["output_buffer"]->set(createInputOutputBuffer(RT_FORMAT_UNSIGNED_BYTE4, m_width, m_height));

  //Set the G Buffer
  m_context["G_buffer"]->set(createInputOutputBuffer(RT_FORMAT_FLOAT3, m_width, m_height));

  //Set the Stencil buffer
  m_context["stencil_buffer"]->set(createInputOutputBuffer(RT_FORMAT_INT, m_width, m_height));

  //Set the Busy buffer
  m_context["busy_buffer"]->set(createInputOutputBuffer(RT_FORMAT_INT, m_width, m_height));

  // Set the temp_buffer
  m_context["temp_buffer"]->set(createOutputBuffer(RT_FORMAT_FLOAT3, m_width, m_height) );
  
  //Set the color buffer
  m_context["color_buffer"]->set(createInputOutputBuffer(RT_FORMAT_UNSIGNED_BYTE4,m_width,m_height));

  //Set the tiled buffer
  m_context["tiled_buffer"]->set(createOutputBuffer(RT_FORMAT_UNSIGNED_BYTE4,m_width,m_height));

  //Set the gradient buffers
  m_context["gradient_buffer"]->set(createInputOutputBuffer1D(RT_FORMAT_FLOAT3, 256));

  //Set the leaf_tile_indices buffer
  //m_context["leaf_tiles_indices"]->set(m_context->createBuffer(RT_BUFFER_INPUT_OUTPUT,RT_FORMAT_UNSIGNED_INT2,256));
  m_context["leaf_tile_indices"]->set(createInputOutputBuffer1D(RT_FORMAT_UNSIGNED_INT2, 256));

  //Set the leaf_tile_sizes buffer
  m_context["leaf_tile_sizes"]->set(createInputOutputBuffer1D(RT_FORMAT_UNSIGNED_INT, 256));

  //Set the parent tile buffers
  m_context["parent_tile_indices"]->set(createInputOutputBuffer1D(RT_FORMAT_UNSIGNED_INT2, 256));
  m_context["parent_tile_sizes"]->set(createInputOutputBuffer1D(RT_FORMAT_UNSIGNED_INT, 256)); 

  //Set the weighted average color buffers
  m_context["wixi_buffer"]->set(createInputOutputBuffer(RT_FORMAT_FLOAT3, m_width,m_height)); 
  m_context["wi_buffer"]->set(createInputOutputBuffer(RT_FORMAT_FLOAT, m_width, m_height)); 

  //Set the gaussian buffers
  m_context["gaussian_x_buffer"]->set(createInputOutputBuffer(RT_FORMAT_UNSIGNED_BYTE4, m_width, m_height));
  m_context["gaussian_y_buffer"]->set(createInputOutputBuffer(RT_FORMAT_UNSIGNED_BYTE4, m_width, m_height));


  //Set the deep buffer??
  //m_context["deep_buffer"]->set(createInputOutputBuffer(RT_FORMAT_USER,m_width, m_height)); 
  Buffer db = createInputOutputBuffer(RT_FORMAT_USER, m_width, m_height, sizeof(List));
  db->setElementSize(sizeof(List));
  m_context["deep_buffer"]->set(db);
  

  //Set the variance buffer : 1 for leaf, and another for parent
  m_context["variance_buffer"]->set(createInputOutputBuffer1D(RT_FORMAT_FLOAT, 256));
  m_context["parent_variance_buffer"]->set(createInputOutputBuffer1D(RT_FORMAT_FLOAT, 256)); 

  // Pinhole Camera ray gen and exception program
  std::string         ptx_path = ptxpath( "whitted", "pinhole_camera.cu" );
  m_context->setRayGenerationProgram( Pinhole, m_context->createProgramFromPTXFile( ptx_path, "pinhole_camera" ) );
  m_context->setExceptionProgram(     Pinhole, m_context->createProgramFromPTXFile( ptx_path, "exception" ) );

  //Abhinav's stuff begins
  //Adding of entry points 

  //Reconstruction
  m_context->setRayGenerationProgram(Reconstruction, m_context->createProgramFromPTXFile(ptx_path, "reconstruct"));
  m_context->setExceptionProgram(Reconstruction, m_context->createProgramFromPTXFile( ptx_path, "exception" ) );

  //Parent
  m_context->setRayGenerationProgram(Parent, m_context->createProgramFromPTXFile(ptx_path, "parent"));
  m_context->setExceptionProgram(Parent, m_context->createProgramFromPTXFile( ptx_path, "exception" ) );

  //Leaf
  m_context->setRayGenerationProgram(Leaf, m_context->createProgramFromPTXFile(ptx_path, "leaf"));
  m_context->setExceptionProgram(Leaf, m_context->createProgramFromPTXFile( ptx_path, "exception" ) );

  //Gradient
  m_context->setRayGenerationProgram(Gradient, m_context->createProgramFromPTXFile(ptx_path, "gradient"));
  m_context->setExceptionProgram(Gradient, m_context->createProgramFromPTXFile( ptx_path, "exception" ) );


  //Abhinav's stuff ends

  // Adaptive Pinhole Camera ray gen and exception program
  ptx_path = ptxpath( "whitted", "adaptive_pinhole_camera.cu" );
  m_context->setRayGenerationProgram( AdaptivePinhole, m_context->createProgramFromPTXFile( ptx_path, "pinhole_camera" ) );
  m_context->setExceptionProgram(     AdaptivePinhole, m_context->createProgramFromPTXFile( ptx_path, "exception" ) );


  // Miss program
  Program miss_program = m_context->createProgramFromPTXFile( ptxpath( "whitted", "constantbg.cu" ), "miss" );
  m_context->setMissProgram( 0, miss_program );

  m_context["bad_color"]->setFloat( 0.0f, 1.0f, 0.0f );
  m_context["bg_color"]->setFloat( make_float3( 0.34f, 0.55f, 0.85f ) ); //Blue sky
  //m_context["bg_color"]->setFloat( make_float3(0.0f, 0.0f, 0.0f)); //Black
  //m_context["bg_color"]->setFloat( make_float3(1.0f, 1.0f, 1.0f)); //white

  // Lights
  BasicLight lights[] = {
    { make_float3( 60.0f, 40.0f, 0.0f ), make_float3( 1.0f, 1.0f, 1.0f ), 1 }
  };

  Buffer light_buffer = m_context->createBuffer(RT_BUFFER_INPUT);
  light_buffer->setFormat(RT_FORMAT_USER);
  light_buffer->setElementSize(sizeof(BasicLight));
  light_buffer->setSize( sizeof(lights)/sizeof(lights[0]) );
  memcpy(light_buffer->map(), lights, sizeof(lights));
  light_buffer->unmap();

  m_context["lights"]->set(light_buffer);

  // Set up camera
  camera_data = InitialCameraData( make_float3( 8.0f, 2.0f, -4.0f ), // eye
                                   make_float3( 4.0f, 2.3f, -4.0f ), // lookat
                                   make_float3( 0.0f, 1.0f,  0.0f ), // up
                                   60.0f );                          // vfov

  m_context["eye"]->setFloat( make_float3( 0.0f, 0.0f, 0.0f ) );
  m_context["U"]->setFloat( make_float3( 0.0f, 0.0f, 0.0f ) );
  m_context["V"]->setFloat( make_float3( 0.0f, 0.0f, 0.0f ) );
  m_context["W"]->setFloat( make_float3( 0.0f, 0.0f, 0.0f ) );

  // Variance buffers
  Buffer variance_sum_buffer = m_context->createBuffer( RT_BUFFER_INPUT_OUTPUT | RT_BUFFER_GPU_LOCAL,
                                                       RT_FORMAT_FLOAT4,
                                                       m_width, m_height );
  memset( variance_sum_buffer->map(), 0, m_width*m_height*sizeof(float4) );
  variance_sum_buffer->unmap();
  m_context["variance_sum_buffer"]->set( variance_sum_buffer );

  Buffer variance_sum2_buffer = m_context->createBuffer( RT_BUFFER_INPUT_OUTPUT | RT_BUFFER_GPU_LOCAL,
                                                        RT_FORMAT_FLOAT4,
                                                        m_width, m_height );
  memset( variance_sum2_buffer->map(), 0, m_width*m_height*sizeof(float4) );
  variance_sum2_buffer->unmap();
  m_context["variance_sum2_buffer"]->set( variance_sum2_buffer );

  // Sample count buffer
  Buffer num_samples_buffer = m_context->createBuffer( RT_BUFFER_INPUT_OUTPUT | RT_BUFFER_GPU_LOCAL,
                                                      RT_FORMAT_UNSIGNED_INT,
                                                      m_width, m_height );
  memset( num_samples_buffer->map(), 0, m_width*m_height*sizeof(unsigned int) );
  num_samples_buffer->unmap();
  m_context["num_samples_buffer"]->set( num_samples_buffer);

  // RNG seed buffer
  m_rnd_seeds = m_context->createBuffer( RT_BUFFER_INPUT_OUTPUT | RT_BUFFER_GPU_LOCAL,
                                               RT_FORMAT_UNSIGNED_INT,
                                               m_width, m_height );
  m_context["rnd_seeds"]->set( m_rnd_seeds );
  genRndSeeds( m_width, m_height );

  // Populate scene hierarchy
  createGeometry();
  //createSimpleGeometry();
  //createComplexGeometry();

  // Prepare to run
  m_context->validate();
  m_context->compile();

  //Set the root tile properties to initialize it
  root_tile.tile_size = 512;
  root_tile.north_east = root_tile.north_west = root_tile.south_east = root_tile.south_west = NULL;
  root_tile.parent = NULL;
  root_tile.centre_index.x = root_tile.centre_index.y = 256;

  //Set rand_tile_index to 0
  rand_tile_index = 0;

  leaf_tiles = root_tile.get_leaf_tiles(leaf_tiles);
  
  printf ("Initial leaf_tiles size = %d \n", leaf_tiles.size());

  while(leaf_tiles.size() < 200)
  {
	  for(int i=0;i<leaf_tiles.size();i++)
	  {
		  leaf_tiles.at(i)->splitTile();
	  }
	  leaf_tiles.clear();
	  leaf_tiles = root_tile.get_leaf_tiles(leaf_tiles);
	  printf("Now, leaf tiles = %d \n", leaf_tiles.size());
  }

  //Add leaf tiles to the pq
  for(int i=0;i<leaf_tiles.size();i++)
  {
	  leaf_pq.push(leaf_tiles.at(i));
	  leaf_tiles.at(i)->is_leaf_tile = true; 
  }

  //leaf_tiles.clear();

  //Populate parent tiles
  parent_tiles.clear();
  parent_tiles = root_tile.calculate_descendants(parent_tiles);
  printf("Parents = %d \n", parent_tiles.size());
  
  for(int i=0;i<parent_tiles.size();i++)
  {
	  parent_tiles.at(i)->is_leaf_tile = false; 
  }

  float s = 666;
  const float* six = &s;
  m_context["number_of_parent_tiles"]->set1fv(six);

}

Buffer WhittedScene::getOutputBuffer()
{
  return m_context["output_buffer"]->getBuffer();
}

// Return whether we processed the key or not
bool WhittedScene::keyPressed(unsigned char key, int x, int y)
{
  switch (key)
  {
  case 'a':
    m_adaptive_aa = !m_adaptive_aa;
    m_camera_changed = true;
    //GLUTDisplay::setContinuousMode( m_adaptive_aa ? GLUTDisplay::CDProgressive : GLUTDisplay::CDNone );
	GLUTDisplay::setContinuousMode(GLUTDisplay::CDProgressive);
	return true;
  }
  return false;
}


void WhittedScene::trace( const RayGenCameraData& camera_data )
{
  if ( m_camera_changed ) {
    m_frame_number = 0u;
    m_camera_changed = false;
  }

  m_context["eye"]->setFloat( camera_data.eye );
  m_context["U"]->setFloat( camera_data.U );
  m_context["V"]->setFloat( camera_data.V );
  m_context["W"]->setFloat( camera_data.W );
  m_context["frame_number"]->setUint( m_frame_number++ );

  Buffer buffer = m_context["output_buffer"]->getBuffer();
  RTsize buffer_width, buffer_height;
  buffer->getSize( buffer_width, buffer_height );


  
  
  //===
  if(m_adaptive_aa)
  {
	  //printf("\nAdaptive!");
	  t = clock();
	  /*
	  m_context->launch( getEntryPoint(),
                   static_cast<unsigned int>(buffer_width),
                   static_cast<unsigned int>(buffer_height) );
	  */
	  m_context->launch(getEntryPoint(), static_cast<unsigned int>(256u), static_cast<unsigned int>(256u));
  
	  t = clock() - t;
	  cpu_time = clock();
	  cpu_time = clock() - cpu_time;
	  GLUTDisplay::setText((float)t/CLOCKS_PER_SEC,(float)cpu_time/CLOCKS_PER_SEC);

  }
  else
  {
	  t = clock();
	  
	  //Shoot
	  m_context->launch(getEntryPoint(), static_cast<unsigned int>(256u), static_cast<unsigned int>(75u));
	  //Parent
	  m_context->launch(Parent, static_cast<unsigned int>(parent_tiles.size()), static_cast<unsigned int>(1u));
	  //Leaf
	  m_context->launch(Leaf, static_cast<unsigned int>(256u), static_cast<unsigned int>(1u));
	  //Gradient
	 // m_context->launch(Gradient, static_cast<unsigned int>(256u), static_cast<unsigned int>(1u));
	  //Reconstruct
	  m_context->launch(Reconstruction,static_cast<unsigned int>(512u), static_cast<unsigned int>(512u)); 
	  

	  t = clock() - t;
	  //printf("Dispatch time : %d clicks \t %f seconds \n",t, ((float)t)/CLOCKS_PER_SEC);

	  //GLUTDisplay::setText((float)t/CLOCKS_PER_SEC);

	  cpu_time = clock();

	  //printf("leaf = %d \n", leaf_tiles.size()); 

	  /*

	  for(int i=0;i<2;i++) //Do more than one retiling per CPU dispatch
	  {
		  doRetiling();
	  }
	  */
	  
	  //drawTiles();
	  retiling_new(1);
	  //retiling();
	  //drawTiles();
	  //doRetiling();
	  //clearStencil();
	  //retiling();
	  
	  cpu_time = clock() - cpu_time;
	  GLUTDisplay::setText((float)t/CLOCKS_PER_SEC,(float)cpu_time/CLOCKS_PER_SEC);


  }
  //Don't edit above this
  //printf("Doing retiling \n");
  //doRetiling();
  
}


void WhittedScene::doResize( unsigned int width, unsigned int height )
{
  // output buffer handled in SampleScene::resize
  m_context["variance_sum_buffer"]->getBuffer()->setSize( width, height );
  m_context["variance_sum2_buffer"]->getBuffer()->setSize( width, height );
  m_context["num_samples_buffer"]->getBuffer()->setSize( width, height );
  m_context["rnd_seeds"]->getBuffer()->setSize( width, height );
  genRndSeeds( width, height );
}

void WhittedScene::createSimpleGeometry()
{
	//Create just one rectangle geometry
	std::string pgram_ptx( ptxpath( "whitted", "parallelogram.cu" ) );
  Geometry parallelogram = m_context->createGeometry();
  parallelogram->setPrimitiveCount( 1u );
  parallelogram->setBoundingBoxProgram( m_context->createProgramFromPTXFile( pgram_ptx, "bounds" ) );
  parallelogram->setIntersectionProgram( m_context->createProgramFromPTXFile( pgram_ptx, "intersect" ) );
  float3 anchor = make_float3( -16.0f, 0.01f, -8.0f );
  float3 v1 = make_float3( 32.0f, 0.0f, 0.0f );
  float3 v2 = make_float3( 0.0f, 0.0f, 16.0f );
  float3 normal = cross( v1, v2 );
  normal = normalize( normal );
  float d = dot( normal, anchor );
  v1 *= 1.0f/dot( v1, v1 );
  v2 *= 1.0f/dot( v2, v2 );
  float4 plane = make_float4( normal, d );
  parallelogram["plane"]->setFloat( plane );
  parallelogram["v1"]->setFloat( v1 );
  parallelogram["v2"]->setFloat( v2 );
  parallelogram["anchor"]->setFloat( anchor );

    // Checker material for floor
  Program check_ch = m_context->createProgramFromPTXFile( ptxpath( "whitted", "checker.cu" ), "closest_hit_radiance" );
  Program check_ah = m_context->createProgramFromPTXFile( ptxpath( "whitted", "checker.cu" ), "any_hit_shadow" );
  Material floor_matl = m_context->createMaterial();
  floor_matl->setClosestHitProgram( 0, check_ch );
  floor_matl->setAnyHitProgram( 1, check_ah );

  floor_matl["Kd1"]->setFloat( 0.8f, 0.3f, 0.15f);
  floor_matl["Ka1"]->setFloat( 0.8f, 0.3f, 0.15f);
  floor_matl["Ks1"]->setFloat( 0.0f, 0.0f, 0.0f);
  floor_matl["Kd2"]->setFloat( 0.8f, 0.3f, 0.15f);
  floor_matl["Ka2"]->setFloat( 0.8f, 0.3f, 0.15f);
  floor_matl["Ks2"]->setFloat( 0.0f, 0.0f, 0.0f);
  
  //floor_matl["Kd2"]->setFloat( 0.9f, 0.85f, 0.05f);
  //floor_matl["Ka2"]->setFloat( 0.9f, 0.85f, 0.05f);
  //floor_matl["Ks2"]->setFloat( 0.0f, 0.0f, 0.0f);
  floor_matl["inv_checker_size"]->setFloat( 32.0f, 16.0f, 1.0f );
  floor_matl["phong_exp1"]->setFloat( 0.0f );
  floor_matl["phong_exp2"]->setFloat( 0.0f );
  floor_matl["reflectivity1"]->setFloat( 0.0f, 0.0f, 0.0f);
  floor_matl["reflectivity2"]->setFloat( 0.0f, 0.0f, 0.0f);

  //Create Gis
  std::vector<GeometryInstance> gis;
  gis.push_back( m_context->createGeometryInstance( parallelogram, &floor_matl,  &floor_matl+1 ) );

  // Place all in group
  GeometryGroup geometrygroup = m_context->createGeometryGroup();
  geometrygroup->setChildCount( static_cast<unsigned int>(gis.size()) );
  geometrygroup->setChild( 0, gis[0] );

  geometrygroup->setAcceleration( m_context->createAcceleration("NoAccel","NoAccel") );

  m_context["top_object"]->set( geometrygroup );
  m_context["top_shadower"]->set( geometrygroup );

}

void WhittedScene::createGeometry()
{
  // Create glass sphere geometry
  std::string shell_ptx( ptxpath( "whitted", "sphere_shell.cu" ) ); 
  Geometry glass_sphere = m_context->createGeometry();
  glass_sphere->setPrimitiveCount( 1u );
  glass_sphere->setBoundingBoxProgram( m_context->createProgramFromPTXFile( shell_ptx, "bounds" ) );
  glass_sphere->setIntersectionProgram( m_context->createProgramFromPTXFile( shell_ptx, "intersect" ) );
  glass_sphere["center"]->setFloat( 4.0f, 2.3f, -4.0f );
  glass_sphere["radius1"]->setFloat( 0.96f );
  glass_sphere["radius2"]->setFloat( 1.0f );
  
  // Metal sphere geometry
  std::string sphere_ptx( ptxpath( "whitted", "sphere.cu" ) ); 
  Geometry metal_sphere = m_context->createGeometry();
  metal_sphere->setPrimitiveCount( 1u );
  metal_sphere->setBoundingBoxProgram( m_context->createProgramFromPTXFile( sphere_ptx, "bounds" ) );
  metal_sphere->setIntersectionProgram( m_context->createProgramFromPTXFile( sphere_ptx, "robust_intersect" ) );
  metal_sphere["sphere"]->setFloat( 2.0f, 1.5f, -2.5f, 1.0f );

  // Floor geometry
  std::string pgram_ptx( ptxpath( "whitted", "parallelogram.cu" ) );
  Geometry parallelogram = m_context->createGeometry();
  parallelogram->setPrimitiveCount( 1u );
  parallelogram->setBoundingBoxProgram( m_context->createProgramFromPTXFile( pgram_ptx, "bounds" ) );
  parallelogram->setIntersectionProgram( m_context->createProgramFromPTXFile( pgram_ptx, "intersect" ) );
  float3 anchor = make_float3( -16.0f, 0.01f, -8.0f );
  float3 v1 = make_float3( 32.0f, 0.0f, 0.0f );
  float3 v2 = make_float3( 0.0f, 0.0f, 16.0f );
  float3 normal = cross( v1, v2 );
  normal = normalize( normal );
  float d = dot( normal, anchor );
  v1 *= 1.0f/dot( v1, v1 );
  v2 *= 1.0f/dot( v2, v2 );
  float4 plane = make_float4( normal, d );
  parallelogram["plane"]->setFloat( plane );
  parallelogram["v1"]->setFloat( v1 );
  parallelogram["v2"]->setFloat( v2 );
  parallelogram["anchor"]->setFloat( anchor );

  // Glass material
  Program glass_ch = m_context->createProgramFromPTXFile( ptxpath( "whitted", "glass.cu" ), "closest_hit_radiance" );
  Program glass_ah = m_context->createProgramFromPTXFile( ptxpath( "whitted", "glass.cu" ), "any_hit_shadow" );
  Material glass_matl = m_context->createMaterial();
  glass_matl->setClosestHitProgram( 0, glass_ch );
  glass_matl->setAnyHitProgram( 1, glass_ah );

  glass_matl["importance_cutoff"]->setFloat( 1e-2f );
  glass_matl["cutoff_color"]->setFloat( 0.034f, 0.055f, 0.085f );
  glass_matl["fresnel_exponent"]->setFloat( 3.0f );
  glass_matl["fresnel_minimum"]->setFloat( 0.1f );
  glass_matl["fresnel_maximum"]->setFloat( 1.0f );
  glass_matl["refraction_index"]->setFloat( 1.4f );
  glass_matl["refraction_color"]->setFloat( 1.0f, 1.0f, 1.0f );
  glass_matl["reflection_color"]->setFloat( 1.0f, 1.0f, 1.0f );
  glass_matl["refraction_maxdepth"]->setInt( 10 );
  glass_matl["reflection_maxdepth"]->setInt( 5 );
  float3 extinction = make_float3(.83f, .83f, .83f);
  glass_matl["extinction_constant"]->setFloat( log(extinction.x), log(extinction.y), log(extinction.z) );
  glass_matl["shadow_attenuation"]->setFloat( 0.6f, 0.6f, 0.6f );

  // Metal material
  Program phong_ch = m_context->createProgramFromPTXFile( ptxpath( "whitted", "phong.cu" ), "closest_hit_radiance" );
  Program phong_ah = m_context->createProgramFromPTXFile( ptxpath( "whitted", "phong.cu" ), "any_hit_shadow" );

  Material metal_matl = m_context->createMaterial();
  metal_matl->setClosestHitProgram( 0, phong_ch );
  metal_matl->setAnyHitProgram( 1, phong_ah );
  metal_matl["Ka"]->setFloat( 0.2f, 0.5f, 0.5f );
  metal_matl["Kd"]->setFloat( 0.2f, 0.7f, 0.8f );
  metal_matl["Ks"]->setFloat( 0.9f, 0.9f, 0.9f );
  metal_matl["phong_exp"]->setFloat( 64 );
  metal_matl["reflectivity"]->setFloat( 0.5f,  0.5f,  0.5f);

  // Checker material for floor
  Program check_ch = m_context->createProgramFromPTXFile( ptxpath( "whitted", "checker.cu" ), "closest_hit_radiance" );
  Program check_ah = m_context->createProgramFromPTXFile( ptxpath( "whitted", "checker.cu" ), "any_hit_shadow" );
  Material floor_matl = m_context->createMaterial();
  floor_matl->setClosestHitProgram( 0, check_ch );
  floor_matl->setAnyHitProgram( 1, check_ah );

  floor_matl["Kd1"]->setFloat( 0.8f, 0.3f, 0.15f);
  floor_matl["Ka1"]->setFloat( 0.8f, 0.3f, 0.15f);
  floor_matl["Ks1"]->setFloat( 0.0f, 0.0f, 0.0f);
  floor_matl["Kd2"]->setFloat( 0.9f, 0.85f, 0.05f);
  floor_matl["Ka2"]->setFloat( 0.9f, 0.85f, 0.05f);
  floor_matl["Ks2"]->setFloat( 0.0f, 0.0f, 0.0f);
  floor_matl["inv_checker_size"]->setFloat( 32.0f, 16.0f, 1.0f );
  floor_matl["phong_exp1"]->setFloat( 0.0f );
  floor_matl["phong_exp2"]->setFloat( 0.0f );
  floor_matl["reflectivity1"]->setFloat( 0.0f, 0.0f, 0.0f);
  floor_matl["reflectivity2"]->setFloat( 0.0f, 0.0f, 0.0f);

  // Create GIs for each piece of geometry
  std::vector<GeometryInstance> gis;
  gis.push_back( m_context->createGeometryInstance( glass_sphere, &glass_matl, &glass_matl+1 ) );
  gis.push_back( m_context->createGeometryInstance( metal_sphere,  &metal_matl,  &metal_matl+1 ) );
  gis.push_back( m_context->createGeometryInstance( parallelogram, &floor_matl,  &floor_matl+1 ) );

  // Place all in group
  GeometryGroup geometrygroup = m_context->createGeometryGroup();
  geometrygroup->setChildCount( static_cast<unsigned int>(gis.size()) );
  geometrygroup->setChild( 0, gis[0] );
  geometrygroup->setChild( 1, gis[1] );
  geometrygroup->setChild( 2, gis[2] );
  geometrygroup->setAcceleration( m_context->createAcceleration("NoAccel","NoAccel") );

  m_context["top_object"]->set( geometrygroup );
  m_context["top_shadower"]->set( geometrygroup );
}

void WhittedScene::createComplexGeometry()
{
  // Create glass sphere geometry
  std::string shell_ptx( ptxpath( "whitted", "sphere_shell.cu" ) ); 
  Geometry glass_sphere = m_context->createGeometry();
  glass_sphere->setPrimitiveCount( 1u );
  glass_sphere->setBoundingBoxProgram( m_context->createProgramFromPTXFile( shell_ptx, "bounds" ) );
  glass_sphere->setIntersectionProgram( m_context->createProgramFromPTXFile( shell_ptx, "intersect" ) );
  glass_sphere["center"]->setFloat( 4.0f, 2.3f, -4.0f );
  glass_sphere["radius1"]->setFloat( 0.96f );
  glass_sphere["radius2"]->setFloat( 1.0f );
  glass_sphere["center"]->setFloat( 4.3f, 2.5f, -5.0f );
  glass_sphere["radius1"]->setFloat( 0.96f );
  glass_sphere["radius2"]->setFloat( 1.0f );

  Geometry glass_sphere_1 = m_context->createGeometry();
  glass_sphere_1->setPrimitiveCount( 1u );
  glass_sphere_1->setBoundingBoxProgram( m_context->createProgramFromPTXFile( shell_ptx, "bounds" ) );
  glass_sphere_1->setIntersectionProgram( m_context->createProgramFromPTXFile( shell_ptx, "intersect" ) );
  glass_sphere_1["center"]->setFloat( 14.0f, 2.3f, -4.0f );
  glass_sphere_1["radius1"]->setFloat( 0.96f );
  glass_sphere_1["radius2"]->setFloat( 1.0f );

  //100 glass spheres
  Geometry glass_spheres[200];
  
  for(int i=0;i<200;i++)
  {
	  float x, y, z;
	  x = static_cast<float> (rand())/static_cast<float> (RAND_MAX);
	  x *= 25.0f;

	  y = static_cast<float> (rand())/static_cast<float> (RAND_MAX);
	  y *= 25.0f;

	  z = static_cast<float> (rand())/static_cast<float> (RAND_MAX);
	  z *= 25.0f;

	  glass_spheres[i] = m_context->createGeometry();
	  glass_spheres[i]->setPrimitiveCount(1u);
	  glass_spheres[i]->setBoundingBoxProgram(m_context->createProgramFromPTXFile(shell_ptx, "bounds"));
	  glass_spheres[i]->setIntersectionProgram(m_context->createProgramFromPTXFile(shell_ptx, "intersect"));
	  //glass_spheres[i]["center"]->setFloat(0.1*i, 0.2*i, -1.0*i);
	  glass_spheres[i]["center"]->setFloat(x,y,z); 
	  glass_spheres[i]["radius1"]->setFloat(0.96f);
	  glass_spheres[i]["radius2"]->setFloat(1.0f); 
  }
  


  // Metal sphere geometry
  std::string sphere_ptx( ptxpath( "whitted", "sphere.cu" ) ); 
  Geometry metal_sphere = m_context->createGeometry();
  metal_sphere->setPrimitiveCount( 1u );
  metal_sphere->setBoundingBoxProgram( m_context->createProgramFromPTXFile( sphere_ptx, "bounds" ) );
  metal_sphere->setIntersectionProgram( m_context->createProgramFromPTXFile( sphere_ptx, "robust_intersect" ) );
  metal_sphere["sphere"]->setFloat( 2.0f, 1.5f, -2.5f, 1.0f );

  // Floor geometry
  std::string pgram_ptx( ptxpath( "whitted", "parallelogram.cu" ) );
  Geometry parallelogram = m_context->createGeometry();
  parallelogram->setPrimitiveCount( 1u );
  parallelogram->setBoundingBoxProgram( m_context->createProgramFromPTXFile( pgram_ptx, "bounds" ) );
  parallelogram->setIntersectionProgram( m_context->createProgramFromPTXFile( pgram_ptx, "intersect" ) );
  float3 anchor = make_float3( -106.0f, 0.01f, -80.0f );
  float3 v1 = make_float3( 500.0f, 0.0f, 0.0f );
  float3 v2 = make_float3( 0.0f, 0.0f, 170.0f );
  float3 normal = cross( v1, v2 );
  normal = normalize( normal );
  float d = dot( normal, anchor );
  v1 *= 1.0f/dot( v1, v1 );
  v2 *= 1.0f/dot( v2, v2 );
  float4 plane = make_float4( normal, d );
  parallelogram["plane"]->setFloat( plane );
  parallelogram["v1"]->setFloat( v1 );
  parallelogram["v2"]->setFloat( v2 );
  parallelogram["anchor"]->setFloat( anchor );

  // Glass material
  Program glass_ch = m_context->createProgramFromPTXFile( ptxpath( "whitted", "glass.cu" ), "closest_hit_radiance" );
  Program glass_ah = m_context->createProgramFromPTXFile( ptxpath( "whitted", "glass.cu" ), "any_hit_shadow" );
  Material glass_matl = m_context->createMaterial();
  glass_matl->setClosestHitProgram( 0, glass_ch );
  glass_matl->setAnyHitProgram( 1, glass_ah );

  glass_matl["importance_cutoff"]->setFloat( 1e-2f );
  glass_matl["cutoff_color"]->setFloat( 0.034f, 0.055f, 0.085f );
  glass_matl["fresnel_exponent"]->setFloat( 3.0f );
  glass_matl["fresnel_minimum"]->setFloat( 0.1f );
  glass_matl["fresnel_maximum"]->setFloat( 1.0f );
  glass_matl["refraction_index"]->setFloat( 1.4f );
  glass_matl["refraction_color"]->setFloat( 1.0f, 1.0f, 1.0f );
  glass_matl["reflection_color"]->setFloat( 1.0f, 1.0f, 1.0f );
  glass_matl["refraction_maxdepth"]->setInt( 10 );
  glass_matl["reflection_maxdepth"]->setInt( 5 );
  float3 extinction = make_float3(.83f, .83f, .83f);
  glass_matl["extinction_constant"]->setFloat( log(extinction.x), log(extinction.y), log(extinction.z) );
  glass_matl["shadow_attenuation"]->setFloat( 0.6f, 0.6f, 0.6f );

  // Metal material
  Program phong_ch = m_context->createProgramFromPTXFile( ptxpath( "whitted", "phong.cu" ), "closest_hit_radiance" );
  Program phong_ah = m_context->createProgramFromPTXFile( ptxpath( "whitted", "phong.cu" ), "any_hit_shadow" );

  Material metal_matl = m_context->createMaterial();
  metal_matl->setClosestHitProgram( 0, phong_ch );
  metal_matl->setAnyHitProgram( 1, phong_ah );
  metal_matl["Ka"]->setFloat( 0.2f, 0.5f, 0.5f );
  metal_matl["Kd"]->setFloat( 0.2f, 0.7f, 0.8f );
  metal_matl["Ks"]->setFloat( 0.9f, 0.9f, 0.9f );
  metal_matl["phong_exp"]->setFloat( 64 );
  metal_matl["reflectivity"]->setFloat( 0.5f,  0.5f,  0.5f);

  // Checker material for floor
  Program check_ch = m_context->createProgramFromPTXFile( ptxpath( "whitted", "checker.cu" ), "closest_hit_radiance" );
  Program check_ah = m_context->createProgramFromPTXFile( ptxpath( "whitted", "checker.cu" ), "any_hit_shadow" );
  Material floor_matl = m_context->createMaterial();
  floor_matl->setClosestHitProgram( 0, check_ch );
  floor_matl->setAnyHitProgram( 1, check_ah );

  floor_matl["Kd1"]->setFloat( 0.8f, 0.3f, 0.15f);
  floor_matl["Ka1"]->setFloat( 0.8f, 0.3f, 0.15f);
  floor_matl["Ks1"]->setFloat( 0.0f, 0.0f, 0.0f);
  floor_matl["Kd2"]->setFloat( 0.9f, 0.85f, 0.05f);
  floor_matl["Ka2"]->setFloat( 0.9f, 0.85f, 0.05f);
  floor_matl["Ks2"]->setFloat( 0.0f, 0.0f, 0.0f);
  floor_matl["inv_checker_size"]->setFloat( 32.0f, 16.0f, 1.0f );
  floor_matl["phong_exp1"]->setFloat( 0.0f );
  floor_matl["phong_exp2"]->setFloat( 0.0f );
  floor_matl["reflectivity1"]->setFloat( 0.0f, 0.0f, 0.0f);
  floor_matl["reflectivity2"]->setFloat( 0.0f, 0.0f, 0.0f);

  // Create GIs for each piece of geometry
  std::vector<GeometryInstance> gis;
  gis.push_back( m_context->createGeometryInstance( glass_sphere_1, &glass_matl, &glass_matl+1 ) );
  gis.push_back( m_context->createGeometryInstance( glass_sphere, &glass_matl, &glass_matl+1 ) );
  gis.push_back( m_context->createGeometryInstance( metal_sphere,  &metal_matl,  &metal_matl+1 ) );
  gis.push_back( m_context->createGeometryInstance( parallelogram, &floor_matl,  &floor_matl+1 ) );

  
  for(int i=0;i<200;i++)
  {
	  if(i<=100)
	  {
		  gis.push_back(m_context->createGeometryInstance(glass_spheres[i], &glass_matl, &glass_matl+1));
	  }
	  else
	  {
		   gis.push_back(m_context->createGeometryInstance(glass_spheres[i], &metal_matl, &metal_matl+1));
	  }
	  
	  //gis.push_back(m_context->createGeometryInstance(glass_spheres[i], &glass_matl, &glass_matl+1));
  }
  

  // Place all in group
  GeometryGroup geometrygroup = m_context->createGeometryGroup();
  geometrygroup->setChildCount( static_cast<unsigned int>(gis.size()) );
  /*
  geometrygroup->setChild( 0, gis[0] );
  geometrygroup->setChild( 1, gis[1] );
  geometrygroup->setChild( 2, gis[2] );
  geometrygroup->setChild( 3, gis[3] );
  */
  
  for(int i=0;i<204;i++)
  {
	  geometrygroup->setChild(static_cast<unsigned int>(i), gis[i]); 
  }
  

  //geometrygroup->setAcceleration( m_context->createAcceleration("NoAccel","NoAccel") );
  geometrygroup->setAcceleration(m_context->createAcceleration("MedianBvh", "Bvh"));

  m_context["top_object"]->set( geometrygroup );
  m_context["top_shadower"]->set( geometrygroup );

}

void WhittedScene::drawTiles()
{
	void* color_data_ptr = m_context["color_buffer"]->getBuffer()->map();
	optix::uchar4* color_data = (optix::uchar4*)color_data_ptr; 

	void* output_data_ptr = m_context["output_buffer"]->getBuffer()->map();
	optix::uchar4* output_data = (optix::uchar4*) output_data_ptr;

	void* gradient_data_ptr = m_context["gradient_buffer"]->getBuffer()->map();
	optix::float3* gradient_data = (optix::float3*)gradient_data_ptr;

	void* stencil_data_ptr = m_context["stencil_buffer"]->getBuffer()->map();
	optix::int1* stencil_data = (optix::int1*)stencil_data_ptr;

	for(int i=0;i<leaf_tiles.size();i++)
	{
		optix::float3 gradient = gradient_data[i];
		//leaf_tiles.at(i)->draw_tile_gradient(color_data, output_data, gradient); 
		leaf_tiles.at(i)->drawTile(color_data, output_data, stencil_data); 
	}

	m_context["color_buffer"]->getBuffer()->unmap();
	m_context["output_buffer"]->getBuffer()->unmap();
	m_context["gradient_buffer"]->getBuffer()->unmap();
	m_context["stencil_buffer"]->getBuffer()->unmap();
}

void WhittedScene::clearStencil()
{

	unsigned int width, height;
	m_context["output_buffer"]->getBuffer()->getSize(width,height); 
    root_tile.centre_index.x = width/2;
    root_tile.centre_index.y = height/2;
	
	
	leaf_tiles.clear();
	leaf_tiles = root_tile.get_leaf_tiles(leaf_tiles);


	void* stencil_data_ptr = m_context["stencil_buffer"]->getBuffer()->map();
	optix::int1* stencil_data = (optix::int1*)stencil_data_ptr;

	for(int i=0;i<leaf_tiles.size();i++)
	{
		leaf_tiles.at(i)->clear_stencil(stencil_data);
	}

	m_context["stencil_buffer"]->getBuffer()->unmap();
}

void WhittedScene::retiling_new(int number_of_retiles)
{
	//Perform all the mappings
	//
	//
	void* output_data = m_context["output_buffer"]->getBuffer()->map();
	optix::uchar4* op_data = (optix::uchar4*) output_data;

	void* col_data = m_context["color_buffer"]->getBuffer()->map();
	optix::uchar4* color_data = (optix::uchar4*)col_data; 
	
	void* variance_data_void = m_context["variance_buffer"]->getBuffer()->map();
	float* variance_data = (float*) variance_data_void;

	void* parent_variance = m_context["parent_variance_buffer"]->getBuffer()->map();
	float* parent_variance_data = (float*) parent_variance; 

	for(int i=0;i<number_of_retiles;i++)
	{
		//Merge
	//
	//
	Tile* min_error_tile;
	float min_error = 99999.99f;

	for(int i=0;i<parent_tiles.size();i++)
	{
		parent_tiles.at(i)->tile_variance = parent_variance_data[i];
		if( parent_tiles.at(i)->tile_size <= 64u && //Make sure tiles don't get too big 
			parent_tiles.at(i)->tile_variance* parent_tiles.at(i)->tile_size < min_error)
		{
			min_error = parent_tiles.at(i)->tile_variance* parent_tiles.at(i)->tile_size;
			min_error_tile = parent_tiles.at(i);
		}

	}

	//Remove the children of the min_error_tile from leaf tiles, and place min_error_til in leaf_tiles
	leaf_tiles.erase(std::remove(leaf_tiles.begin(), leaf_tiles.end(), min_error_tile->north_east), leaf_tiles.end());
	leaf_tiles.erase(std::remove(leaf_tiles.begin(), leaf_tiles.end(), min_error_tile->north_west), leaf_tiles.end());
	leaf_tiles.erase(std::remove(leaf_tiles.begin(), leaf_tiles.end(), min_error_tile->south_east), leaf_tiles.end());
	leaf_tiles.erase(std::remove(leaf_tiles.begin(), leaf_tiles.end(), min_error_tile->south_west), leaf_tiles.end());
	
	//Now, make the min_error_tile a leaf tile by calling mergeTile
	min_error_tile->mergeTile(); //Merge the parent tile with least error

	//Now add min_error_tile to leaf_tiles, as it now has no children
	leaf_tiles.push_back(min_error_tile); 
	
	//Calculate variance of min_error_tile as it is a newly added leaf tile
	min_error_tile->calculate_variance(color_data, op_data);

	//Remove min_error_tile from the parent tiles too
	parent_tiles.erase(std::remove(parent_tiles.begin(), parent_tiles.end(), min_error_tile), parent_tiles.end());

	//Add grandpa tile to parent tile if it has 4 children exactly.
	Tile* grandpa = min_error_tile->parent;
	if(grandpa->north_east->is_leaf_tile &&
		grandpa->north_west->is_leaf_tile &&
		grandpa->south_east->is_leaf_tile &&
		grandpa->south_west->is_leaf_tile)
	{
		parent_tiles.push_back(grandpa); 
	}

	//Split
	//
	//
	Tile* max_error_tile;
	max_error_tile = leaf_tiles.front();
	float max_error = 0.0f; 

	for(int i=0;i<leaf_tiles.size();i++)
	{
		//leaf_tiles.at(i)->calculate_variance(color_data,op_data);
		leaf_tiles.at(i)->tile_variance = variance_data[i];
		if(leaf_tiles.at(i)->tile_variance * leaf_tiles.at(i)->tile_size >= max_error)
		{
			max_error = leaf_tiles.at(i)->tile_size*leaf_tiles.at(i)->tile_variance;
			max_error_tile = leaf_tiles.at(i);
		}
	}

	max_error_tile->splitTile(); //Split leaf tile with max error
	
	//Add the newly created leaf tiles to the vector
	leaf_tiles.push_back(max_error_tile->north_east);
	leaf_tiles.push_back(max_error_tile->north_west);
	leaf_tiles.push_back(max_error_tile->south_east);
	leaf_tiles.push_back(max_error_tile->south_west);

	//Add the max_error_tile as a parent tile
	parent_tiles.push_back(max_error_tile); 
	max_error_tile->number_of_descendants = 4;

	//Remove the max_error_tile from leaf tiles as it's no longer a leaf tile 
	leaf_tiles.erase(std::remove(leaf_tiles.begin(), leaf_tiles.end(), max_error_tile), leaf_tiles.end());
	
	Tile* grandparent = max_error_tile->parent;
	parent_tiles.erase(std::remove(parent_tiles.begin(), parent_tiles.end(), grandparent), parent_tiles.end());
	
	}//for loop ends 

	//Perform all the unmappings
	//
	//
	m_context["parent_variance_buffer"]->getBuffer()->unmap();
	m_context["variance_buffer"]->getBuffer()->unmap();
	m_context["output_buffer"]->getBuffer()->unmap();
	m_context["color_buffer"]->getBuffer()->unmap();

	//Place leaf and parent data into appropriate buffers
	//
	//

	//Fill the leaf tiles buffer
	void* leaf_data = m_context["leaf_tile_indices"]->getBuffer()->map();
	optix::uint2* leaf_indices = (optix::uint2*)leaf_data;

	void* leaf_tile_size_data = m_context["leaf_tile_sizes"]->getBuffer()->map();
	unsigned int* leaf_sizes = (unsigned int*)leaf_tile_size_data;

	for(int i=0;i<leaf_tiles.size();i++)
	{
		leaf_indices[i] = leaf_tiles.at(i)->centre_index;
		leaf_sizes[i] = leaf_tiles.at(i)->tile_size;
	}
	m_context["leaf_tile_sizes"]->getBuffer()->unmap();
	m_context["leaf_tile_indices"]->getBuffer()->unmap();
	
	//Map the parent buffers 
	void* parent_data = m_context["parent_tile_indices"]->getBuffer()->map();
	optix::uint2* parent_indices = (optix::uint2*) parent_data;

	void* parent_size_data = m_context["parent_tile_sizes"]->getBuffer()->map();
	unsigned int* parent_sizes = (unsigned int*)parent_size_data;

	for(int i=0; i<parent_tiles.size();i++)
	{
		parent_indices[i] = parent_tiles.at(i)->centre_index;
		parent_sizes[i] = parent_tiles.at(i)->tile_size; 
	}

	m_context["parent_tile_sizes"]->getBuffer()->unmap();
	m_context["parent_tile_indices"]->getBuffer()->unmap();
}

void WhittedScene::retiling()
{
	//Perform all the mappings
	//
	//
	void* output_data = m_context["output_buffer"]->getBuffer()->map();
	optix::uchar4* op_data = (optix::uchar4*) output_data;

	void* col_data = m_context["color_buffer"]->getBuffer()->map();
	optix::uchar4* color_data = (optix::uchar4*)col_data; 
	
	void* variance_data_void = m_context["variance_buffer"]->getBuffer()->map();
	float* variance_data = (float*) variance_data_void;

	void* parent_variance = m_context["parent_variance_buffer"]->getBuffer()->map();
	float* parent_variance_data = (float*) parent_variance; 

	//Merge
	//
	//
	Tile* min_error_tile;
	float min_error = 99999.99f;

	for(int i=0;i<parent_tiles.size();i++)
	{
		parent_tiles.at(i)->tile_variance = parent_variance_data[i];
		if( parent_tiles.at(i)->tile_size <= 64u && //Make sure tiles don't get too big 
			parent_tiles.at(i)->tile_variance* parent_tiles.at(i)->tile_size < min_error)
		{
			min_error = parent_tiles.at(i)->tile_variance* parent_tiles.at(i)->tile_size;
			min_error_tile = parent_tiles.at(i);
		}

	}

	//Remove the children of the min_error_tile from leaf tiles, and place min_error_til in leaf_tiles
	leaf_tiles.erase(std::remove(leaf_tiles.begin(), leaf_tiles.end(), min_error_tile->north_east), leaf_tiles.end());
	leaf_tiles.erase(std::remove(leaf_tiles.begin(), leaf_tiles.end(), min_error_tile->north_west), leaf_tiles.end());
	leaf_tiles.erase(std::remove(leaf_tiles.begin(), leaf_tiles.end(), min_error_tile->south_east), leaf_tiles.end());
	leaf_tiles.erase(std::remove(leaf_tiles.begin(), leaf_tiles.end(), min_error_tile->south_west), leaf_tiles.end());
	
	//Now, make the min_error_tile a leaf tile by calling mergeTile
	min_error_tile->mergeTile(); //Merge the parent tile with least error

	//Now add min_error_tile to leaf_tiles, as it now has no children
	leaf_tiles.push_back(min_error_tile); 
	
	//Calculate variance of min_error_tile as it is a newly added leaf tile
	min_error_tile->calculate_variance(color_data, op_data);

	//Remove min_error_tile from the parent tiles too
	parent_tiles.erase(std::remove(parent_tiles.begin(), parent_tiles.end(), min_error_tile), parent_tiles.end());

	//Add grandpa tile to parent tile if it has 4 children exactly.
	Tile* grandpa = min_error_tile->parent;
	if(grandpa->north_east->is_leaf_tile &&
		grandpa->north_west->is_leaf_tile &&
		grandpa->south_east->is_leaf_tile &&
		grandpa->south_west->is_leaf_tile)
	{
		parent_tiles.push_back(grandpa); 
	}

	//Split
	//
	//
	Tile* max_error_tile;
	max_error_tile = leaf_tiles.front();
	float max_error = 0.0f; 

	for(int i=0;i<leaf_tiles.size();i++)
	{
		//leaf_tiles.at(i)->calculate_variance(color_data,op_data);
		leaf_tiles.at(i)->tile_variance = variance_data[i];
		if(leaf_tiles.at(i)->tile_variance * leaf_tiles.at(i)->tile_size >= max_error)
		{
			max_error = leaf_tiles.at(i)->tile_size*leaf_tiles.at(i)->tile_variance;
			max_error_tile = leaf_tiles.at(i);
		}
	}

	max_error_tile->splitTile(); //Split leaf tile with max error
	
	//Add the newly created leaf tiles to the vector
	leaf_tiles.push_back(max_error_tile->north_east);
	leaf_tiles.push_back(max_error_tile->north_west);
	leaf_tiles.push_back(max_error_tile->south_east);
	leaf_tiles.push_back(max_error_tile->south_west);

	//Add the max_error_tile as a parent tile
	parent_tiles.push_back(max_error_tile); 
	max_error_tile->number_of_descendants = 4;

	//Remove the max_error_tile from leaf tiles as it's no longer a leaf tile 
	leaf_tiles.erase(std::remove(leaf_tiles.begin(), leaf_tiles.end(), max_error_tile), leaf_tiles.end());
	
	Tile* grandparent = max_error_tile->parent;
	parent_tiles.erase(std::remove(parent_tiles.begin(), parent_tiles.end(), grandparent), parent_tiles.end());

	//Perform all the unmappings
	//
	//
	m_context["parent_variance_buffer"]->getBuffer()->unmap();
	m_context["variance_buffer"]->getBuffer()->unmap();
	m_context["output_buffer"]->getBuffer()->unmap();
	m_context["color_buffer"]->getBuffer()->unmap();

	//Place leaf and parent data into appropriate buffers
	//
	//

	//Fill the leaf tiles buffer
	void* leaf_data = m_context["leaf_tile_indices"]->getBuffer()->map();
	optix::uint2* leaf_indices = (optix::uint2*)leaf_data;

	void* leaf_tile_size_data = m_context["leaf_tile_sizes"]->getBuffer()->map();
	unsigned int* leaf_sizes = (unsigned int*)leaf_tile_size_data;

	for(int i=0;i<leaf_tiles.size();i++)
	{
		leaf_indices[i] = leaf_tiles.at(i)->centre_index;
		leaf_sizes[i] = leaf_tiles.at(i)->tile_size;
	}
	m_context["leaf_tile_sizes"]->getBuffer()->unmap();
	m_context["leaf_tile_indices"]->getBuffer()->unmap();
	
	//Map the parent buffers 
	void* parent_data = m_context["parent_tile_indices"]->getBuffer()->map();
	optix::uint2* parent_indices = (optix::uint2*) parent_data;

	void* parent_size_data = m_context["parent_tile_sizes"]->getBuffer()->map();
	unsigned int* parent_sizes = (unsigned int*)parent_size_data;

	for(int i=0; i<parent_tiles.size();i++)
	{
		parent_indices[i] = parent_tiles.at(i)->centre_index;
		parent_sizes[i] = parent_tiles.at(i)->tile_size; 
	}

	m_context["parent_tile_sizes"]->getBuffer()->unmap();
	m_context["parent_tile_indices"]->getBuffer()->unmap();

}

void WhittedScene::doRetiling()
{
	//m_context["output_buffer"]->getBuffer();
	//Tile root_tile;
	//Buffer opbuffer =  m_context["output_buffer"]->getBuffer();
	/**/
	//unsigned int width, height;
	//m_context["output_buffer"]->getBuffer()->getSize(width,height); 
    //root_tile.centre_index.x = width/2;
    //root_tile.centre_index.y = height/2;
	
	//printf("centre = %d %d \n", root_tile.centre_index.x, root_tile.centre_index.y);
	//root_tile.tile_size = 512;
	//root_tile.north_east = root_tile.north_west = root_tile.south_east = root_tile.south_west = NULL;
	//root_tile.parent = NULL;
	/**/
	//RTsize w1,h1;
	//m_context["output_buffer"]->getBuffer()->getSize(w1,h1);
	
	void* data = m_context["output_buffer"]->getBuffer()->map();
	optix::uchar4* op_data = (optix::uchar4*)data;

	void* col_data = m_context["color_buffer"]->getBuffer()->map();
	optix::uchar4* color_data = (optix::uchar4*)col_data; 

	void* stencil_data_ptr = m_context["stencil_buffer"]->getBuffer()->map();
	optix::int1* stencil_data = (optix::int1*)stencil_data_ptr;

	//root_tile.calculate_variance(color_data, op_data);

	//Uncomment?
	//leaf_tiles.clear();
	//leaf_tiles = root_tile.get_leaf_tiles(leaf_tiles);

	//Populate parent tiles==================================uncomment enxt two lines 
	//parent_tiles.clear();
	//parent_tiles = root_tile.calculate_descendants(parent_tiles);

	//printf("Befor Merge leaf = %d \t parent = %d \n", leaf_tiles.size(), parent_tiles.size());

	//Merge leaf parent tile with least error
	float min_error = 1000000.0f;
	Tile* min_error_tile;
	float max_error = 0.0f;
	Tile* max_error_tile;

	//===============
	for(int i=0;i<parent_tiles.size();i++)
	{
		parent_tiles.at(i)->calculate_variance(color_data, op_data);
		if( parent_tiles.at(i)->tile_size <= 64u && //Make sure tiles don't get too big 
			parent_tiles.at(i)->tile_variance* parent_tiles.at(i)->tile_size < min_error)
		{
			min_error = parent_tiles.at(i)->tile_variance* parent_tiles.at(i)->tile_size;
			min_error_tile = parent_tiles.at(i);
		}
		//printf("Parent tile num desc = %d \n", parent_tiles.at(i)->number_of_descendants);
	}
	
	//Remove the children of the min_error_tile from leaf tiles, and place min_error_til in leaf_tiles
	leaf_tiles.erase(std::remove(leaf_tiles.begin(), leaf_tiles.end(), min_error_tile->north_east), leaf_tiles.end());
	leaf_tiles.erase(std::remove(leaf_tiles.begin(), leaf_tiles.end(), min_error_tile->north_west), leaf_tiles.end());
	leaf_tiles.erase(std::remove(leaf_tiles.begin(), leaf_tiles.end(), min_error_tile->south_east), leaf_tiles.end());
	leaf_tiles.erase(std::remove(leaf_tiles.begin(), leaf_tiles.end(), min_error_tile->south_west), leaf_tiles.end());
	
	//Now, make the min_error_tile a leaf tile by calling mergeTile
	min_error_tile->mergeTile(); //Merge the parent tile with least error

	//Now add min_error_tile to leaf_tiles, as it now has no children
	leaf_tiles.push_back(min_error_tile); 

	//Remove min_error_tile from the parent tiles too
	parent_tiles.erase(std::remove(parent_tiles.begin(), parent_tiles.end(), min_error_tile), parent_tiles.end());

	//Add grandpa tile to parent tile if it has 4 children exactly.
	Tile* grandpa = min_error_tile->parent;
	if(grandpa->north_east->is_leaf_tile &&
		grandpa->north_west->is_leaf_tile &&
		grandpa->south_east->is_leaf_tile &&
		grandpa->south_west->is_leaf_tile)
	{
		parent_tiles.push_back(grandpa); 
	}

	//Repopulate the leaf tiles .... uncomment next 2 lines?
	//leaf_tiles.clear();
	//leaf_tiles = root_tile.get_leaf_tiles(leaf_tiles);

	//printf("After Merge leaf = %d \t parent = %d \n", leaf_tiles.size(), parent_tiles.size());

	for(int i=0;i<leaf_tiles.size();i++)
	{
		leaf_tiles.at(i)->calculate_variance(color_data,op_data);
		if(leaf_tiles.at(i)->tile_variance * leaf_tiles.at(i)->tile_size >= max_error)
		{
			max_error = leaf_tiles.at(i)->tile_size*leaf_tiles.at(i)->tile_variance;
			max_error_tile = leaf_tiles.at(i);
		}
	}

	//printf("max_error_tile =  %u %u \t size = %u \n", 
		//max_error_tile->centre_index.x, max_error_tile->centre_index.y, max_error_tile->tile_size);
	max_error_tile->splitTile(); //Split leaf tile with max error
	
	//leaf_tiles.clear();
	//leaf_tiles = root_tile.get_leaf_tiles(leaf_tiles);

	
	//Add the newly created leaf tiles to the vector
	leaf_tiles.push_back(max_error_tile->north_east);
	leaf_tiles.push_back(max_error_tile->north_west);
	leaf_tiles.push_back(max_error_tile->south_east);
	leaf_tiles.push_back(max_error_tile->south_west);

	//Add the max_error_til as a parent tile
	parent_tiles.push_back(max_error_tile); 
	max_error_tile->number_of_descendants = 4;

	//Remove the max_error_tile from leaf tiles as it's no longer a leaf tile 
	leaf_tiles.erase(std::remove(leaf_tiles.begin(), leaf_tiles.end(), max_error_tile), leaf_tiles.end());
	

	//printf("After Split leaf = %d \t parent = %d \n", leaf_tiles.size(), parent_tiles.size());

	Tile* grandparent = max_error_tile->parent;
	parent_tiles.erase(std::remove(parent_tiles.begin(), parent_tiles.end(), grandparent), parent_tiles.end());
	//printf("After Split leaf = %d \t parent = %d \n \n", leaf_tiles.size(), parent_tiles.size());
	//printf("Grandparent = %d \n\n", grandparent->number_of_descendants);
	//printf("nod = %d \n", parent_tiles.back()->number_of_descendants);
	//printf("===\n");

	//printf("Parent = %d \n",parent_tiles.size());
	//printf("Split  = %d \n", leaf_tiles.size()); 

	//===============

	//printf("Number of leaf tiles after merging and splitting together = %d\n", leaf_tiles.size());

	//Finally, draw all leaf tiles
	/*
	for(int i=0;i<leaf_tiles.size();i++)
	{
		leaf_tiles.at(i)->drawTile(color_data,op_data, stencil_data);
	}
	*/
	
	//Get the leaf tiles again. Avoid case where child is split and its parent is merged
	
	//leaf_tiles.clear();
	//leaf_tiles = root_tile.get_leaf_tiles(leaf_tiles);
	
	m_context["output_buffer"]->getBuffer()->unmap();
	m_context["color_buffer"]->getBuffer()->unmap();
	m_context["stencil_buffer"]->getBuffer()->unmap();

	//Fill the leaf tiles buffer
	void* leaf_data = m_context["leaf_tile_indices"]->getBuffer()->map();
	optix::uint2* leaf_indices = (optix::uint2*)leaf_data;

	void* leaf_tile_size_data = m_context["leaf_tile_sizes"]->getBuffer()->map();
	unsigned int* leaf_sizes = (unsigned int*)leaf_tile_size_data;

	for(int i=0;i<leaf_tiles.size();i++)
	{
		leaf_indices[i] = leaf_tiles.at(i)->centre_index;
		leaf_sizes[i] = leaf_tiles.at(i)->tile_size;
		//printf("Leaf tile index copied is %u %u \t Size = %u \t %d \n", leaf_indices[i].x, leaf_indices[i].y, leaf_sizes[i], i);
	}
	//printf("===========\n");
	m_context["leaf_tile_sizes"]->getBuffer()->unmap();
	m_context["leaf_tile_indices"]->getBuffer()->unmap();
	
	//Map the parent buffers 
	void* parent_data = m_context["parent_tile_indices"]->getBuffer()->map();
	optix::uint2* parent_indices = (optix::uint2*) parent_data;

	void* parent_size_data = m_context["parent_tile_sizes"]->getBuffer()->map();
	unsigned int* parent_sizes = (unsigned int*)parent_size_data;

	for(int i=0; i<parent_tiles.size();i++)
	{
		parent_indices[i] = parent_tiles.at(i)->centre_index;
		parent_sizes[i] = parent_tiles.at(i)->tile_size; 
	}

	m_context["parent_tile_sizes"]->getBuffer()->unmap();
	m_context["parent_tile_indices"]->getBuffer()->unmap();




	float s = (float)parent_tiles.size();
	const float* fs = &s;
	m_context["number_of_parent_tiles"]->set1fv(fs); 

	//printf("Number of leaf tiles after merging = %d\n", leaf_tiles.size());

	//print the variance buffer over here to see if all is ok
	/*
	void* variance_data_void = m_context["variance_buffer"]->getBuffer()->map();
	float* variance_data = (float*) variance_data_void;

	for(int i=0; i<leaf_tiles.size();i++)
	{
		printf("Variance = %f \t Tile = %d \n", variance_data[i], i);
	}
	
	m_context["variance_buffer"]->getBuffer()->unmap();
	*/
 
	
}

//-----------------------------------------------------------------------------
//
// Main driver
//
//-----------------------------------------------------------------------------


void printUsageAndExit( const std::string& argv0, bool doExit = true )
{
  std::cerr
    << "Usage  : " << argv0 << " [options]\n"
    << "App options:\n"
    << "  -h  | --help                               Print this usage message\n"
    << "  -A  | --adaptive-off                       Turn off adaptive AA\n"
    << std::endl;
  GLUTDisplay::printUsage();

  std::cerr
    << "App keystrokes:\n"
    << "  a Toggles adaptive pixel sampling on and off\n"
    << std::endl;

  if ( doExit ) exit(1);
}

int main( int argc, char** argv )
{
  GLUTDisplay::init( argc, argv );
  bool adaptive_aa = true;
  for ( int i = 1; i < argc; ++i ) {
    std::string arg( argv[i] );
    if ( arg == "--adaptive-off" || arg == "-A" ) {
      adaptive_aa = false;
    } else if ( arg == "--help" || arg == "-h" ) {
      printUsageAndExit( argv[0] );
    } else {
      std::cerr << "Unknown option: '" << arg << "'\n";
      printUsageAndExit( argv[0] );
    }
  }

  if( !GLUTDisplay::isBenchmark() ) printUsageAndExit( argv[0], false );


  while(true)
  {

  try {
    WhittedScene scene;
    scene.setAdaptiveAA( adaptive_aa );
    GLUTDisplay::run( "WhittedScene", &scene, adaptive_aa ? GLUTDisplay::CDProgressive : GLUTDisplay::CDNone );
  } catch( Exception& e ){
    sutilReportError( e.getErrorString().c_str() );
    exit(1);
  }
  }

  return 0;
}
