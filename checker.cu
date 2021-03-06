
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
#include <optixu/optixu_math_namespace.h>
#include "phong.h" 

using namespace optix;

rtDeclareVariable(float3,       Kd1, , );
rtDeclareVariable(float3,       Kd2, , );
rtDeclareVariable(float3,       Ka1, , );
rtDeclareVariable(float3,       Ka2, , );
rtDeclareVariable(float3,       Ks1, , );
rtDeclareVariable(float3,       Ks2, , );
rtDeclareVariable(float3,       reflectivity1, , );
rtDeclareVariable(float3,       reflectivity2, , );
rtDeclareVariable(float,        phong_exp1, , );
rtDeclareVariable(float,        phong_exp2, , );
rtDeclareVariable(float3,       inv_checker_size, , );  // Inverse checker height, width and depth in texture space

rtDeclareVariable(float3, texcoord, attribute texcoord, ); 
rtDeclareVariable(float3, geometric_normal, attribute geometric_normal, ); 
rtDeclareVariable(float3, shading_normal, attribute shading_normal, ); 


RT_PROGRAM void any_hit_shadow()
{
  phongShadowed();
}


RT_PROGRAM void closest_hit_radiance()
{
  float3 Kd, Ka, Ks, reflectivity;
  float  phong_exp;

  float3 t  = texcoord * inv_checker_size;
  t.x = floorf(t.x);
  t.y = floorf(t.y);
  t.z = floorf(t.z);

  int which_check = ( static_cast<int>( t.x ) +
                      static_cast<int>( t.y ) +
                      static_cast<int>( t.z ) ) & 1;

  if ( which_check ) {
    Kd = Kd1; Ka = Ka1; Ks = Ks1; reflectivity = reflectivity1; phong_exp = phong_exp1;
  } else {
    Kd = Kd2; Ka = Ka2; Ks = Ks2; reflectivity = reflectivity2; phong_exp = phong_exp2;
  }

  float3 world_shading_normal   = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, shading_normal));
  float3 world_geometric_normal = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, geometric_normal));
  float3 ffnormal  = faceforward( world_shading_normal, -ray.direction, world_geometric_normal );
  phongShade( Kd, Ka, Ks, ffnormal, phong_exp, reflectivity );
}
