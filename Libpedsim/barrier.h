#ifndef BARRIER
#define BARRIER

#include <mutex>
#include <atomic>
#include <chrono>

struct Barrier
{
	void wait() 
	{
		++counter_;
		while (!go_.load())
		{
			std::this_thread::sleep_for(interval_);
		}
	}
	
	int get_waiting()
	{
		return counter_.load();
	}

	void release()
	{
		counter_ = 0;
		go_ = true;
	}

	std::atomic<int>
		counter_ = 0;
	std::atomic<bool> 
		go_ = false;
	std::chrono::nanoseconds
		interval_ = std::chrono::nanoseconds(0);

};

#endif
