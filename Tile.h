#pragma  once

#include<cstddef>
#include <optixu/optixpp_namespace.h>
#include <optixu/optixu_math_namespace.h>
#include"helpers.h"
#include<optix_world.h>
#include<math.h>
#include<time.h>

struct XY
{

};

using namespace optix;

struct Tile
{
	Tile* north_west;
	Tile* north_east;
	Tile* south_west;
	Tile* south_east;

	Tile* parent;
	optix::uint2 centre_index; //Stores the centre pixel's index for the tile
	unsigned int tile_size; //Size of the tile dimensions. 8 for a tile that's 8 pxs tall and wide
	float tile_variance; //optix::luminance(variance as a float3) 
	
	int number_of_descendants; //Stores the number of descendatns of each tile

	//methods
	void splitTile(); //If tile variance is > 0.3, it splites the tile 
	void drawTile(optix::uchar4*, optix::uchar4*, optix::int1*); //Draws all the leaf children of this tile if any.
	void draw_tile_gradient(optix::uchar4*, optix::uchar4*, optix::float3); //draw tile with gradient
	Tile* get_max_error_tile(); //returns the maximum variance tile within that tile
	void calculate_variance(optix::uchar4*, optix::uchar4*); //calculates the variance associated with the tile
	bool contains(optix::uint2); //returns true if a point is within this tile. False otherwise
	//void init();//Sets the min and max x and y values
	std::vector<Tile*> get_leaf_tiles(std::vector<Tile*>); //Returns all the leaf tiles 
	std::vector<Tile*> get_parent_tiles(std::vector<Tile*>); //Returns all the parent tiles
	void mergeTile();
	std::vector<Tile*> calculate_descendants(std::vector<Tile*>);
	void clear_stencil(optix::int1*); //Clear the stencil buffer in the tile 
	bool is_leaf_tile; //true if the given tile is a leaf tile 

	bool operator<(const Tile& a)
	{
		return(this->tile_variance < a.tile_variance);
	}

};

Tile* Tile::get_max_error_tile()
{
	if(north_east == NULL &&
		north_west == NULL &&
		south_east == NULL &&
		south_west == NULL)
	{
		return this; //If leaf, return
	}
	else
	{
		Tile *nw,*ne,*sw,*se;
		nw = north_west->get_max_error_tile();
		ne = north_east->get_max_error_tile();
		sw = south_west->get_max_error_tile();
		se = south_east->get_max_error_tile();

		if(nw->tile_variance>= ne->tile_variance
			&& nw->tile_variance >= sw->tile_variance
			&& nw->tile_variance >= se->tile_variance)
		{
			return nw;
		}
		else if (ne->tile_variance >= sw->tile_variance
			&& ne->tile_variance >= se->tile_variance)
		{
			return ne;
		}
		else if (sw->tile_variance>= se->tile_variance)
		{
			return sw;
		}
		else
		{
			return se;
		}

	}
}

void Tile::clear_stencil(optix::int1* stencil_data)
{
	if(north_west == NULL &&
		north_east == NULL &&
		south_west == NULL &&
		south_east == NULL)
	{
		unsigned int half_spacing = tile_size >> 1;
		unsigned int min_x, max_x,min_y,max_y;
		min_x = centre_index.x-half_spacing;
		max_x = centre_index.x+half_spacing;
		min_y = centre_index.y-half_spacing;
		max_y = centre_index.y+half_spacing;

		for(unsigned int i= min_x;i<max_x;++i)
		{
			for(unsigned int j= min_y;j<max_y;++j)
			{
				int index = (int)(i*512+j); 
				if(stencil_data[index].x != 0) //If the pixel is new, paint is as green
				{
					stencil_data[index].x = 0; //Set the stencil value back to zero. 
				}
			}
		}
	}
}

void Tile::draw_tile_gradient(optix::uchar4 * c_data, optix::uchar4* o_data, optix::float3 gradient)
{
	unsigned int half_spacing = tile_size >> 1;
	unsigned int min_x, max_x,min_y,max_y;
	min_x = centre_index.x-half_spacing;
	max_x = centre_index.x+half_spacing;
	min_y = centre_index.y-half_spacing;
	max_y = centre_index.y+half_spacing;

	for(unsigned int i= min_x; i<max_x; ++i)
	{
		for(unsigned int j= min_y; j< max_y; ++j)
		{
			int index = (int)(i*512+j);
			optix::uchar4 old_bytes = c_data[index];
		    optix::float3 color = optix::make_float3(old_bytes.z, old_bytes.y, old_bytes.x)*optix::make_float3(1.f/255.0f);
			/*
			optix::uchar4 new_bytes;
			new_bytes.x = old_bytes.x * gradient.x * 256;
			new_bytes.y = old_bytes.y * gradient.y * 256;
			new_bytes.z = old_bytes.z * gradient.z * 0.256; 
			//new_bytes.z = old_bytes.z;
			new_bytes.w = 255; 
			*/
			
			optix::uchar4 new_bytes;
			new_bytes.x = gradient.x * 256 * 100;
			new_bytes.y = gradient.y * 256 * 100;
			new_bytes.z = gradient.z * 30;// * 0.156;
			new_bytes.w = 255; 
			 
			o_data[index] = new_bytes;
		}
	}


}

void Tile::drawTile(optix::uchar4* c_data, optix::uchar4* o_data, optix::int1* stencil_data)
{
	//calculate_variance(c_data, o_data);
	//splitTile();
	if(north_west == NULL &&
		north_east == NULL &&
		south_west == NULL &&
		south_east == NULL)
	{
		//If the tile is a leaf tile, draw it
		//calculate_variance(c_data, o_data);
		unsigned int half_spacing = tile_size >> 1;
		unsigned int min_x, max_x,min_y,max_y;
		min_x = centre_index.x-half_spacing;
		max_x = centre_index.x+half_spacing;
		min_y = centre_index.y-half_spacing;
		max_y = centre_index.y+half_spacing;

		for(unsigned int i= min_x;i<max_x;++i)
		{
			for(unsigned int j= min_y;j<max_y;++j)
			{
			int index = (int)(i*512+j);
			optix::uchar4 old_bytes = c_data[index];
		    optix::float3 color = optix::make_float3(old_bytes.z, old_bytes.y, old_bytes.x)*optix::make_float3(1.f/255.0f);
			optix::uchar4 new_bytes = old_bytes;
			//Uncomment to include transparency with variance.
			new_bytes.x = old_bytes.x * tile_variance;
			new_bytes.y = old_bytes.y * tile_variance;
			new_bytes.z = old_bytes.z * tile_variance;
			new_bytes.w = old_bytes.w * tile_variance;

			
			if(stencil_data[index].x == 1) //If the pixel is new, paint is as green
			{
				new_bytes.x =  0;
				new_bytes.y = 255;
				new_bytes.z = 0;
				new_bytes.w = 255;

				stencil_data[index].x = 0; //Set the stencil value back to zero. 
			}
			/*
			if(stencil_data[index].x == 2) //If the pixel is centre, paint is as blue
			{
				new_bytes.x =  255;
				new_bytes.y = 0;
				new_bytes.z = 255;
				new_bytes.w = 255;

				//stencil_data[index].x = 0; //Set the stencil value back to zero. 
			}


			//Draw the edges of the tile in white. 
			if (i == min_x || i == max_x || j == min_y || j == max_y)
			{
				new_bytes.x = 255;
				new_bytes.y = 255;
				new_bytes.z = 255;
				new_bytes.w = 255;
			}
			*/

			o_data[index] = new_bytes;
			}
			
		}
		
		//printf("%f = tile_varaince\n", tile_variance);
	}
	else
	{
		//Else draw its children
		north_west->drawTile(c_data, o_data, stencil_data);
		north_east->drawTile(c_data, o_data, stencil_data);
		south_west->drawTile(c_data, o_data, stencil_data);
		south_east->drawTile(c_data, o_data, stencil_data);
	
		//south_east->drawTile(c_data, o_data);
		//south_west->drawTile(c_data, o_data);
		//north_west->drawTile(c_data, o_data);
		//north_east->drawTile(c_data, o_data);
	}

}

void Tile::splitTile()
{
	if(/*tile_variance > 0.3 &&*/ /*tile_size> 8u*/ true)
	{
		Tile* n_w = new Tile();
		Tile* n_e = new Tile();
		Tile* s_w = new Tile();
		Tile* s_e = new Tile();
		
		north_east = n_e;
		north_west = n_w;
		south_east = s_e;
		south_west = s_w;

		unsigned int quarter_spacing = tile_size>>2;
		north_west->centre_index.x = centre_index.x - quarter_spacing;
		north_west->centre_index.y = centre_index.y + quarter_spacing;

		north_east->centre_index.x = centre_index.x + quarter_spacing;
		north_east->centre_index.y = centre_index.y + quarter_spacing;

		south_east->centre_index.x = centre_index.x + quarter_spacing;
		south_east->centre_index.y = centre_index.y - quarter_spacing;

		south_west->centre_index.x = centre_index.x - quarter_spacing;
		south_west->centre_index.y = centre_index.y - quarter_spacing;

		north_west->tile_size = tile_size/2;
		north_east->tile_size = tile_size/2;
		south_east->tile_size = tile_size/2;
		south_west->tile_size = tile_size/2;

		north_west->parent = this;
		north_east->parent = this;
		south_west->parent = this;
		south_east->parent = this;

		north_west->is_leaf_tile = true;
		north_east->is_leaf_tile = true;
		south_west->is_leaf_tile = true;
		south_east->is_leaf_tile = true; 

		is_leaf_tile = false;
	}
	else
	{
		printf("Didn't split because of min size 8u \n");
	}

}

void Tile::mergeTile()
{
	if(north_west != NULL)
	{
		delete north_west;
		north_west = NULL;
	}
	if(north_east != NULL)
	{
		delete north_east;
		north_east = NULL;
	}
	if(south_west != NULL)
	{
		delete south_west;
		south_west = NULL;
	}
	if(south_east != NULL)
	{
		delete south_east;
		south_east = NULL;
	}
	is_leaf_tile = true; 
}

std::vector<Tile*> Tile::calculate_descendants(std::vector<Tile*> tiles)
{
	//Calculates number of descendents, and if it is 4, pushes to vector, as its a parent tile
	//Returns a vector of parent tiles 
	if(north_west == NULL &&
		north_east == NULL &&
		south_west == NULL &&
		south_east == NULL) //if tile is a leaf tile
	{
		number_of_descendants = 0;
	}
	else
	{
		tiles = north_west->calculate_descendants(tiles); 
		tiles = north_east->calculate_descendants(tiles); 
		tiles = south_west->calculate_descendants(tiles); 
		tiles = south_east->calculate_descendants(tiles); 
		number_of_descendants = north_west->number_of_descendants+
								north_east->number_of_descendants+
								south_west->number_of_descendants+
								south_east->number_of_descendants+
								4;
	
	}

	//printf("NOD = %d \n", number_of_descendants);

	if(number_of_descendants == 4) //If tile has exactly 4 desc. Push to list
	{
		tiles.push_back(this);
	}
	return tiles;
}

std::vector<Tile*> Tile::get_parent_tiles(std::vector<Tile*> tiles)
{
	if(north_west == NULL &&
		north_east == NULL &&
		south_west == NULL &&
		south_east == NULL) //if tile is a leaf tile
	{
		bool parent_exists = false;
		for(int i=0;i<tiles.size();i++)
		{
			if(this->parent == tiles.at(i))
			{
				parent_exists = true;
			}
		}
		if(!parent_exists)
		{
			tiles.push_back(this->parent);
		}
		return tiles;
	}
	else
	{
		tiles = north_west->get_parent_tiles(tiles);
		tiles = north_east->get_parent_tiles(tiles);
		tiles = south_west->get_parent_tiles(tiles);
		tiles = south_east->get_parent_tiles(tiles);
		return tiles;
	}
}

std::vector<Tile*> Tile::get_leaf_tiles(std::vector<Tile*> tiles)
{
	if(north_west == NULL &&
		north_east == NULL &&
		south_west == NULL &&
		south_east == NULL)
	{
		tiles.push_back(this);
		return tiles;
	}
	else
	{
		tiles = north_west->get_leaf_tiles(tiles);
		tiles = north_east->get_leaf_tiles(tiles);
		tiles = south_east->get_leaf_tiles(tiles);
		tiles = south_west->get_leaf_tiles(tiles);
		return tiles;
	}
}

bool Tile::contains(optix::uint2 index)
{
	unsigned int half_spacing = tile_size/2;
	unsigned int min_x, max_x,min_y,max_y;
	min_x = centre_index.x-half_spacing;
	max_x = centre_index.x+half_spacing;
	min_y = centre_index.y-half_spacing;
	max_y = centre_index.y+half_spacing;

	//printf("minx = %u , maxx = %u, miny = %u maxy = %u", min_x, max_x, min_y, max_y);

	if(index.x >= min_x && index.x <= max_x &&
	   index.y >= min_y && index.y <= max_y)
	{
		return true;
	}
	else
	{
		return false;
	}
}

void Tile::calculate_variance(optix::uchar4* color_data, optix::uchar4* op_data)
{
	unsigned int half_spacing = tile_size >> 1;
	unsigned int min_x, max_x,min_y,max_y;
	min_x = centre_index.x-half_spacing;
	max_x = centre_index.x+half_spacing;
	min_y = centre_index.y-half_spacing;
	max_y = centre_index.y+half_spacing;

	float number_of_samples = 0.0;

	optix::float3 m_color;
	m_color.x = m_color.y = m_color.z = 0.0;

	for(unsigned int i= min_x;i<max_x;++i)
	{
		for(unsigned int j = min_y;j<max_y;++j)
		{
			int index = (int)(i*512+j); 
			optix::uchar4 old_bytes = color_data[index];
		    optix::float3 color = optix::make_float3(old_bytes.z, old_bytes.y, old_bytes.x)*optix::make_float3(1.f/255.0f);
			
			m_color += color;
			number_of_samples++;
		}
	}
	m_color /= number_of_samples;
	//printf("%f %f %f = m_color\n", m_color.x, m_color.y, m_color.z);

	optix::uchar4 avg_color;
	avg_color.x = static_cast<unsigned char>(m_color.z * 255.99f);
	avg_color.y = static_cast<unsigned char>(m_color.y * 255.99f);
	avg_color.z = static_cast<unsigned char>(m_color.x * 255.99f);
	avg_color.w = 255u;

	optix::float3 variance; variance.x = variance.y = variance.z = 0.0;
	optix::float3 difference = variance; 
	optix::float3 difference_squared = variance;
	optix::float3 difference_squared_sum = variance;

	for(unsigned int i=min_x;i<max_x; ++i)
	{
		for(unsigned int j=min_y;j<max_y;++j)
		{
			int index = (int)(i*512+j);
			//op_data[index] = avg_color;
			optix::uchar4 old_bytes = color_data[index];
		    optix::float3 color = optix::make_float3(old_bytes.z, old_bytes.y, old_bytes.x)*optix::make_float3(1.f/255.0f);

			difference = color - m_color;
			difference_squared = difference * difference;
			difference_squared_sum += difference_squared;
			
		}
	}

	variance = difference_squared_sum / number_of_samples;
	
	float transparency = optix::luminance(variance);
	//printf("%f \n", transparency);
	
	//transparency *= 9.99;

	//printf("v = %f %f %f \n", variance.x, variance.y, variance.z);

	//Draw the tile along with transparency
	/*
	for(unsigned int i= min_x;i<max_x;++i)
	{
		for(unsigned int j= min_y;j<max_y;++j)
		{
			int index = (int)(i*512+j);
			optix::uchar4 old_bytes = color_data[index];
		    optix::float3 color = optix::make_float3(old_bytes.z, old_bytes.y, old_bytes.x)*optix::make_float3(1.f/255.0f);
			optix::uchar4 new_bytes = old_bytes;
			new_bytes.x = old_bytes.x * transparency;
			new_bytes.y = old_bytes.y * transparency;
			new_bytes.z = old_bytes.z * transparency;
			new_bytes.w = old_bytes.w * transparency;
			op_data[index] = new_bytes;
		}
	}
	*/
	tile_variance = transparency; //Set the tile's variance value;

	//printf("%f = transparency \n", transparency);

	return;
}


//Comparison operator for use by the Priority Queue
bool operator<(const std::shared_ptr<Tile> a, std::shared_ptr<Tile> b)
{
	return (a->tile_variance < b->tile_variance);
}

struct MoreVariance
{
	bool operator()(const Tile* x, const Tile* y) const
	{
		return(x->tile_variance < y->tile_variance);
	}
};

struct LessVariance
{
	bool operator()(const Tile* x, const Tile* y) const
	{
		return(x->tile_variance > y->tile_variance); 
	}
};


struct node
{
	uchar4 color;
	node* next;
	node* previous;
	//uchar4 colors[5];
};

struct List
{
	node* head;
	int head_index;
	int count;
	uchar4 colors[5];
	float gradient_x[5];
	float gradient_y[5]; 
	float gradient_t[5]; 
	float3 extents; 
	clock_t last_sample_time; 

	void add_sample(uchar4); 
	void add_sample(uchar4, float, float, float); 
	void add_sample_simple(uchar4);
	void set_sample_time();
	clock_t get_sample_time();
	uchar4 return_average_color();
	uchar4 weighted_average_color();
	uchar4 return_sample_at_age(int); 
	uchar4 return_front_sample();
	float3 get_gradients(); //returns weighted average x, y, & t gradients.  
	void set_extents(float3); 
	float3 get_extents();
};

__device__ float3 List::get_extents()
{
	return(extents); 
}

__device__ void List::set_extents(float3 ex)
{
	extents = ex;
}

__device__ clock_t List::get_sample_time()
{
	return(last_sample_time); 
}

__device__ void List::set_sample_time()
{
	last_sample_time = clock(); 
}

__device__ uchar4 List::return_sample_at_age(int age)
{
	int index = head_index-age;
	if(index <0)
	{
		index += 5;
	}
	return(colors[index]); 
}

__device__ uchar4 List::return_front_sample()
{
	return(colors[head_index]); 
}

__device__ float3 List::get_gradients()
{
	float3 gradient;
	gradient.x = gradient.y = gradient.z = 0.0f;

	for(int i=0; i< 5;i++)
	{
		int age = (head_index + 5 -i) % 5;
		age = age + 1;

		float w = 1.0/(age);

		gradient += make_float3(gradient_x[i] * w, gradient_y[i] * w, gradient_t[i] * w);

	}

	gradient *= make_float3(1.0f/ 5.0f); 

	return gradient;
}

__device__ uchar4 List::weighted_average_color()
{
	uchar4 avg_color;
	float3 mean_color;
	mean_color.x = mean_color.y = mean_color.z = 0.0f;

	for(int i=0;i<5;i++)
	{
		int age = (head_index + 5 - i) % 5;
		age = age + 1;
		/*
		if(age == 0)
		{
			age = 5;
		}
		*/
		//float w = exp(age * -1);
		float w = 1.0/(age);

		float3 temp_color = make_float3(colors[i].x, colors[i].y, colors[i].z)*make_float3(1.0f/255.99f);
		mean_color += (temp_color * make_float3(w));
	}

	mean_color = mean_color*make_float3(1.0f/5.0f);
	
	avg_color = make_uchar4(static_cast<unsigned char>((mean_color.z)*255.99f), 
                               static_cast<unsigned char>((mean_color.y)*255.99f),  
                               static_cast<unsigned char>((mean_color.x)*255.99f),  
                               255u);

	return avg_color;
}

__device__ uchar4 List::return_average_color()
{
	uchar4 avg_color;
	float3 mean_color;
	mean_color.x = mean_color.y = mean_color.z = 0.0f;

	for(int i=0;i<5;i++)
	{
		float3 temp_color = make_float3(colors[i].x, colors[i].y, colors[i].z)*make_float3(1.0f/255.99f);
		mean_color += temp_color;
	}
	mean_color = mean_color*make_float3(1.0f/5.0f);
	
	avg_color = make_uchar4(static_cast<unsigned char>((mean_color.z)*255.99f), 
                               static_cast<unsigned char>((mean_color.y)*255.99f),  
                               static_cast<unsigned char>((mean_color.x)*255.99f),  
                               255u);
	/*
	avg_color.x = colors[0].x + colors[1].x + colors[2].x +  colors[3].x + colors[4].x;
	avg_color.y = colors[0].y + colors[1].y + colors[2].y +  colors[3].y + colors[4].y;
	avg_color.z = colors[0].z + colors[1].z + colors[2].z +  colors[3].z + colors[4].z;
	avg_color.w = colors[0].w + colors[1].w + colors[2].w +  colors[3].w + colors[4].w;

	avg_color.x /= 5u;
	avg_color.y /= 5u;
	avg_color.z /= 5u;
	avg_color.w /= 5u; 
	*/
	//avg_color = (colors[0] + colors[1] + colors[2] + colors[3] + colors[4])/5;
	//printf("%u %u %u %u \n", avg_color.x, avg_color.y, avg_color.z, avg_color.w);
	return(avg_color); 
}

__device__ void List::add_sample_simple(uchar4 new_color)
{
	/*
	colors[4] = colors[3];
	colors[3] = colors[2];
	colors[2] = colors[1];
	colors[1] = colors[0];
	colors[0] = new_color;
	*/
	int new_index = (head_index + 1 )%5;
	colors[new_index] = new_color;
	head_index = new_index;

}

__device__ void List::add_sample(uchar4 new_color, float g_x, float g_y, float g_t)
{
	int new_index = (head_index + 1)%5;
	colors[new_index] = new_color;
	gradient_x[new_index] = g_x;
	gradient_y[new_index] = g_y;
	gradient_t[new_index] = g_t; 
	head_index = new_index; 
}


 __device__ void List::add_sample(uchar4 new_color)
{
	
	if(count == 5)
	{
		head->previous->color = new_color;
		head = head->previous;
	}
	else if(count == 0)
	{
		node* new_node = (node*)malloc(sizeof(node));
		new_node->color = new_color;
		new_node->next = new_node;
		new_node->previous = new_node;
		head = new_node; 
		count++;
	}
	else
	{
		node* new_node = (node*)malloc(sizeof(node));
		new_node->color = new_color; 
		new_node->next = head;
		new_node->previous = head->previous;
		head->previous->next = new_node;
		head->previous = new_node;
		head = new_node; 
		count++;
	}
}
