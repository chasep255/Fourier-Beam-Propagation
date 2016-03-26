#include <thrust/complex.h>
#include <iostream>
#include <cstdio>
#include <cstring>
#include <string>
#include <cufft.h>
#include <sys/time.h>
#include <thread>
#include "FourierBeamPropagator.cuh"
#include "OpticalElements.cuh"
#include <gtk/gtk.h>
#include <thread>
#include <cstdlib>
#include "DoubleBuffer.hpp"

typedef float real;
typedef thrust::complex<real> complex;

const int CELLS = 1024 * 4;
const real LAMBDA = 500.0e-9;
const real DZ = LAMBDA;
const real CELL_DIM = LAMBDA;
const real MAX_N_CHANGE = 0.005;
const bool ADJUST_STEP = true;
const int RENDER_EVERY = 3;

typedef BiConvexLens<real> OpticalPath;
OpticalPath op(0.001, 0.005, 0.001, 1.3);
fftbpm::BeamPropagator<real, OpticalPath>* prop = nullptr;

inline void cucheck()
{
	cudaError_t err = cudaGetLastError();
	if(err != cudaSuccess)
	{
		throw std::runtime_error(cudaGetErrorString(err));
	}
}

double prof_time()
{
	struct timeval now;
	gettimeofday(&now, NULL);
	return now.tv_sec + 1.0e-6 * now.tv_usec;
}

const int STRIDE = cairo_format_stride_for_width(CAIRO_FORMAT_RGB24, 2 * CELLS);
DoubleBuffer<unsigned char> render_buffer(STRIDE * CELLS);
volatile int total_steps = 0;
volatile real current_z = 0.0;
volatile bool quit_thread = false; 
volatile bool changed = true;
void stepper_thread()
{
	static std::thread t;
	long long count = 0;
	
	cudaSetDevice(prop->getDevice());
	
	while(!quit_thread)
	{
		prop->step(DZ, ADJUST_STEP);
		
		if(count % RENDER_EVERY == 0)
		{
			complex* efld = new complex[CELLS * CELLS];
			prop->getElectricField(efld);
			if(t.joinable())
				t.join();
			t = std::thread([efld]()
			{
				unsigned char* buffer = render_buffer.getRenderBuffer();
				unsigned char* display_buffer = render_buffer.getDisplayBuffer();
				complex* nbuf = new complex[CELLS * CELLS];
				current_z = prop->getZCoordinate();
				total_steps = prop->getStepCount();
				
				#pragma omp parallel for
				for(int y = 0; y < CELLS; y++)
					for(int x = 0; x < CELLS; x++)
						nbuf[y * CELLS + x] = prop->indexOfRefraction((x - CELLS / 2) * CELL_DIM, (y - CELLS / 2) * CELL_DIM, current_z);
				
				#pragma omp parallel for
				for(int y = 0; y < CELLS; y++)
				{
					for(int x = 0; x < CELLS; x++)
					{
						unsigned char val = fmin(255.0, 255.0 * thrust::abs(efld[y * CELLS + x]));
						buffer[y * STRIDE + 4 * x + 0] = 255.0 * (nbuf[y * CELLS + x].real() - 1.0);
						buffer[y * STRIDE + 4 * x + 1] = val;
						buffer[y * STRIDE + 4 * x + 2] = val;
						buffer[y * STRIDE + 4 * x + 3] = 0;
					}
					
					for(int x = CELLS; x < 2 * CELLS - 1; x++)
					{
						buffer[y * STRIDE + 4 * x + 0] = display_buffer[y * STRIDE + 4 * (x + 1) + 0];
						buffer[y * STRIDE + 4 * x + 1] = display_buffer[y * STRIDE + 4 * (x + 1) + 1];
						buffer[y * STRIDE + 4 * x + 2] = display_buffer[y * STRIDE + 4 * (x + 1) + 2];
						buffer[y * STRIDE + 4 * x + 3] = display_buffer[y * STRIDE + 4 * (x + 1) + 3];
					}
					
					unsigned char val = fmin(255.0, 255.0 * thrust::abs(efld[y * CELLS + (CELLS / 2)]));
					buffer[y * STRIDE + 8 * (CELLS - 1) + 0] = 255.0 * (nbuf[y * CELLS + (CELLS / 2)].real() - 1.0);
					buffer[y * STRIDE + 8 * (CELLS - 1) + 1] = val;
					buffer[y * STRIDE + 8 * (CELLS - 1) + 2] = val;
					buffer[y * STRIDE + 8 * (CELLS - 1) + 3] = 0;
				}
				render_buffer.swapBuffers();
				changed = true;
				delete[] nbuf;
				delete[] efld;
			});
		}
		count++;
	}
	if(t.joinable())
		t.join();
}

GtkWidget* drawing_area;
cairo_surface_t* s = nullptr;
gboolean update_surface()
{
	if(!changed)
		return TRUE;
	changed = false;
	if(s) cairo_surface_destroy(s);
	s = cairo_image_surface_create_for_data(render_buffer.getDisplayBuffer(), CAIRO_FORMAT_RGB24, 2 * CELLS, CELLS, STRIDE);
	gtk_widget_queue_draw(GTK_WIDGET(drawing_area));

	return TRUE;
}

double start;
gboolean draw(GtkWidget* widget, cairo_t* cr, gpointer data)
{
	guint width, height;
	
	width = gtk_widget_get_allocated_width(widget);
	height = gtk_widget_get_allocated_height(widget);
	
	cairo_set_source_rgb(cr, 0.0, 0.0, 0.0);
	cairo_rectangle(cr, 0.0, 0.0, width, height);
	cairo_fill(cr);
	
	real scale = fmin(2 * CELLS > width ? (real)width / (2 * CELLS) : 1.0, CELLS > height ? (real)height / CELLS : 1.0);
	
	cairo_scale(cr, scale, scale);
	if(s) cairo_set_source_surface(cr, s, 0.0, 0.0);
	
	cairo_paint(cr);
	cairo_set_source_rgb(cr, 1.0, 0.0, 0.0);
	cairo_scale(cr, 1.0 / scale, 1.0 / scale);
	
	real yoff = 0.0;
	auto cairo_print = [&](const char* text)
	{
		cairo_text_extents_t te;
		cairo_text_extents (cr, text, &te);
		cairo_move_to(cr, 1.0, yoff + te.height + 1.0);
		cairo_show_text(cr, text);
		yoff += te.height + 1.0;
	};
	
	char text[200];
	sprintf(text, "%0.6lf m", current_z);
	cairo_print(text);
	
	sprintf(text, "%d steps", total_steps);
	cairo_print(text);
	
	sprintf(text, "%llu s run time", (unsigned long long)(prof_time()  - start));
	cairo_print(text);
	
	sprintf(text, "%d steps / min", (int)(total_steps / ((prof_time()  - start) / 60.0)));
	cairo_print(text);
	
	sprintf(text, "%0.1lf x %0.1lf um", prop->getPhysicalGridDim() * 1.0e6, prop->getPhysicalGridDim() * 1.0e6);
	cairo_print(text);

	if(LAMBDA < 1.0e-6)
		sprintf(text, "Wavelength %0.1lf nm", LAMBDA * 1.0e9);
	else
		sprintf(text, "Wavelength %0.1lf um", LAMBDA * 1.0e6);
	cairo_print(text);
	return FALSE;
}

int main(int argc, char** argv)
{
	using namespace fftbpm;
	cudaSetDevice(1);
	start = prof_time();
	gtk_init(&argc, &argv);
	
	GtkWidget* window = gtk_window_new(GTK_WINDOW_TOPLEVEL);
	int swidth = gdk_screen_get_height(gdk_screen_get_default());
	int sheight = gdk_screen_get_height(gdk_screen_get_default());
	int width = 2 * CELLS;
	int height = CELLS;
	
	while(width > swidth && height > sheight)
	{
		width /= 2;
		height /= 2;
	}
	
	gtk_window_set_default_size(GTK_WINDOW(window), width, height);
	g_signal_connect(window, "destroy", gtk_main_quit, NULL);
	
	drawing_area = gtk_drawing_area_new();
	gtk_container_add(GTK_CONTAINER(window), drawing_area);
	g_signal_connect(drawing_area, "draw", G_CALLBACK(draw), NULL);

	
	prop = new fftbpm::BeamPropagator<real, OpticalPath>(CELLS, LAMBDA, CELL_DIM, 0.0, op);
	prop->setMaxNChange(MAX_N_CHANGE);
	complex* efld = new complex[CELLS * CELLS];
	for(int y = 0; y < CELLS; y++)
	{
		for(int x = 0; x < CELLS; x++)
		{
			int dx = x - CELLS / 2;
			int dy = y - CELLS / 2;
			efld[y * CELLS + x] = std::exp(-(dx * dx + dy * dy) / 6000.0);
//			if(dx * dx + dy * dy < 100 * 100)
//				efld[y * CELLS + x] = 1.0;
//			else
//				efld[y * CELLS + x] = 0.0;
			
//			int x_prime = x % 100;
//			int y_prime = y % 100;
//			
//			if(x_prime > 20 && x_prime < 80 && y_prime < 80 && y_prime > 20)
//			{
//				if(y_prime > 75 || (x_prime > 47 && x_prime < 53))
//					efld[y * CELLS + x] = 1.0;
//				else
//					efld[y * CELLS + x] = 0.0;
//			}
//			else
//				efld[y * CELLS + x] = 0.0;
//			
//			real dx = CELL_DIM * (x - CELLS / 2);
//			real dy = CELL_DIM * (y - CELLS / 2);
//			real r = sqrt(dx * dx + dy * dy);
//			real f = 0.005;
//			int n = 2 * (sqrt(r * r + f * f) - f) / LAMBDA;
//			efld[y * CELLS + x] = n % 2 && n < 6;
			
//			if(x > 400 && x < 600 && y > 490 && y < 500)
//				efld[y * CELLS + x] = 1.0;
//			else
//				efld[y * CELLS + x] = 0.0;
			
//			efld[y * CELLS + x] = 1.0;
		}
	}
	prop->setElectricField(efld);
	delete[] efld;
	
	gtk_widget_show_all(window);
	
	g_idle_add((GSourceFunc)update_surface, NULL);
	
	std::thread t(stepper_thread);
	
	gtk_main();
	quit_thread = true;
	t.join();
	delete prop;
	return 0;
}
