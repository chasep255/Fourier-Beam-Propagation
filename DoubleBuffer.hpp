#ifndef _DOUBLE_BUFFER_HPP_
#define _DOUBEL_BUFFER_HPP_

#include <mutex>
#include <utility>

template<typename T>
class DoubleBuffer
{
	public:
	
	DoubleBuffer(size_t buffer_size)
	{
		display = new T[buffer_size];
		render = new T[buffer_size];
	}
	
	~DoubleBuffer()
	{
		delete[] display;
		delete[] render;
	}
	
	T* getRenderBuffer()
	{
		std::lock_guard<std::mutex> lg(lock); 
		return render;
	}
	
	T* getDisplayBuffer()
	{
		std::lock_guard<std::mutex> lg(lock); 
		return display;
	}
	
	void swapBuffers()
	{
		std::lock_guard<std::mutex> lg(lock);
		std::swap(render, display);
	}
	
	private:
	std::mutex lock;
	T* display;
	T* render;
};

#endif
