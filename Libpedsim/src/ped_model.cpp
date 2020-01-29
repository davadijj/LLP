//
// pedsim - A microscopic pedestrian simulation system.
// Copyright (c) 2003 - 2014 by Christian Gloor
//
//
// Adapted for Low Level Parallel Programming 2017
//
#include "ped_model.h"
#include "ped_waypoint.h"
#include "ped_model.h"
#include <iostream>
#include <stack>
#include <algorithm>
#include "cuda_testkernel.h"
#include <omp.h>
#include <thread>
#include <chrono>
#include <ctime>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include "../barrier.h"

// Memory leak check with msvc++
#define _CRTDBG_MAP_ALLOC
#include <stdlib.h>
#include <crtdbg.h>
#ifdef _DEBUG
#define new new(_NORMAL_BLOCK, __FILE__, __LINE__)
#endif

/*
std::vector<condition_variable> cv_vec;
std::vector<std::mutex> mutex_vec;
std::vector<bool> bool_vec;
std::vector<thread> threads(0);
*/
std::condition_variable cv_barrier1, cv_barrier2;
std::mutex mut_workers, test1, test2;
std::atomic<bool> finished;
std::vector<thread> threads(0);
int iterations = 0;

//
std::vector<std::mutex*> mutex_0;
std::vector<std::mutex*> mutex_1;
//

Barrier barrier_1, barrier_2;

//void Ped::Model::process_ticks(int i)
void process_ticks(int start, std::vector<Ped::Tagent*> * agents_p, int nr_threads)
{
	std::vector<Ped::Tagent*> agents = *agents_p;
	int chunk_size = agents.size() / nr_threads;
	start = start * chunk_size;
	int stop = start + chunk_size;
	while (!finished.load())
	{
		// Barrier 1
		{
			barrier_1.wait();
		}

		for (int i = start; i < stop; i++)
		{
			agents[i]->computeNextDesiredPosition();
			agents[i]->setX(agents[i]->getDesiredX());
			agents[i]->setY(agents[i]->getDesiredY());
		}
		// Barrier 2
		{
			barrier_2.wait();
		}
	}
}
void process_ticks_mutex(int a, std::vector<Ped::Tagent*> * agents_p, int nr_threads)
{
	int start = a;
	mutex_1[a]->lock();
	std::vector<Ped::Tagent*> agents = *agents_p;
	int chunk_size = agents_p->size() / nr_threads;
	start = start * chunk_size;
	int stop = start + chunk_size;
	while(!finished.load()){
		mutex_0[a]->lock();
		mutex_1[a]->unlock();
		for (int i = start; i < stop; i++)
		{
			(*agents_p)[i]->computeNextDesiredPosition();
			(*agents_p)[i]->setX((*agents_p)[i]->getDesiredX());
			(*agents_p)[i]->setY((*agents_p)[i]->getDesiredY());
		}
		mutex_0[a]->unlock();
		mutex_1[a]->lock();
	}
	mutex_1[a]->unlock();
}


void Ped::Model::setup(std::vector<Ped::Tagent*> agentsInScenario, std::vector<Twaypoint*> destinationsInScenario, IMPLEMENTATION implementation)
{
	
	// Convenience test: does CUDA work on this machine?
	cuda_test();

	// Set 
	agents = std::vector<Ped::Tagent*>(agentsInScenario.begin(), agentsInScenario.end());

	// Set up destinations
	destinations = std::vector<Ped::Twaypoint*>(destinationsInScenario.begin(), destinationsInScenario.end());

	// Sets the chosen implemenation. Standard in the given code is SEQ
	this->implementation = implementation;

	// Set up heatmap (relevant for Assignment 4)
	setupHeatmapSeq();

	//spawn threads 
	finished.store(false);
	if (implementation == 3) {
		for (int i = 0; i < nr_threads; ++i)
		{
			threads.push_back(std::thread(process_ticks, i, &agents, nr_threads));
			/*
			cv_vec.push_back(std::condition_variable());
			mutex_vec.push_back(std::mutex());
			bool_vec.push_back(bool());
			*/
		}
	}
	else if (implementation == 5) {
		for (int i = 0; i < nr_threads; ++i)
		{
			mutex_0.push_back(new std::mutex);
			mutex_0[i]->lock();
			mutex_1.push_back(new std::mutex);
			threads.push_back(std::thread(process_ticks_mutex, i, &agents, nr_threads));
		};
	};
}


std::chrono::nanoseconds nano_secs = 0ns;
void Ped::Model::tick()
{
	// EDIT HERE FOR ASSIGNMENT 1
	// SEQ

	if (implementation == 4) {
		int i = 0;
		for (i = 0; i < agents.size(); i++) {
			agents[i]->computeNextDesiredPosition();
			agents[i]->setX(agents[i]->getDesiredX());
			agents[i]->setY(agents[i]->getDesiredY());
		}
	}
	// Open MP
	else if(implementation == 2) {
		omp_set_num_threads(nr_threads);
		#pragma omp parallel
		{
			int i = 0;
			#pragma omp for private(i)
			for (i = 0; i < agents.size(); i++) {
				agents[i]->computeNextDesiredPosition();
				agents[i]->setX(agents[i]->getDesiredX());
				agents[i]->setY(agents[i]->getDesiredY());
			}
		}
	}
	// cpp threads
	
	else if (implementation == 3) 
	{
		++iterations;
		{
			while (barrier_1.get_waiting() != nr_threads)
				std::this_thread::sleep_for(nano_secs);
			barrier_2.go_ = false;
			barrier_1.release();

			while (barrier_2.get_waiting() != nr_threads)
				std::this_thread::sleep_for(nano_secs);
			barrier_1.go_ = false;
			barrier_2.release();
		}
		//std::cout << iterations << std::endl;
	}
	else if (implementation == 5)
	{
		int i = 0;
		for (i = 0; i < nr_threads; i++) {
			mutex_0[i]->unlock();
			mutex_1[i]->lock();
		}
		for (i = 0; i < nr_threads; i++) {
			mutex_0[i]->lock();
			mutex_1[i]->unlock();
		}
	}
}

////////////
/// Everything below here relevant for Assignment 3.
/// Don't use this for Assignment 1!
///////////////////////////////////////////////

// Moves the agent to the next desired position. If already taken, it will
// be moved to a location close to it.
void Ped::Model::move(Ped::Tagent *agent)
{
	// Search for neighboring agents
	set<const Ped::Tagent *> neighbors = getNeighbors(agent->getX(), agent->getY(), 2);

	// Retrieve their positions
	std::vector<std::pair<int, int> > takenPositions;
	for (std::set<const Ped::Tagent*>::iterator neighborIt = neighbors.begin(); neighborIt != neighbors.end(); ++neighborIt) {
		std::pair<int, int> position((*neighborIt)->getX(), (*neighborIt)->getY());
		takenPositions.push_back(position);
	}

	// Compute the three alternative positions that would bring the agent
	// closer to his desiredPosition, starting with the desiredPosition itself
	std::vector<std::pair<int, int> > prioritizedAlternatives;
	std::pair<int, int> pDesired(agent->getDesiredX(), agent->getDesiredY());
	prioritizedAlternatives.push_back(pDesired);

	int diffX = pDesired.first - agent->getX();
	int diffY = pDesired.second - agent->getY();
	std::pair<int, int> p1, p2;
	if (diffX == 0 || diffY == 0)
	{
		// Agent wants to walk straight to North, South, West or East
		p1 = std::make_pair(pDesired.first + diffY, pDesired.second + diffX);
		p2 = std::make_pair(pDesired.first - diffY, pDesired.second - diffX);
	}
	else {
		// Agent wants to walk diagonally
		p1 = std::make_pair(pDesired.first, agent->getY());
		p2 = std::make_pair(agent->getX(), pDesired.second);
	}
	prioritizedAlternatives.push_back(p1);
	prioritizedAlternatives.push_back(p2);

	// Find the first empty alternative position
	for (std::vector<pair<int, int> >::iterator it = prioritizedAlternatives.begin(); it != prioritizedAlternatives.end(); ++it) {

		// If the current position is not yet taken by any neighbor
		if (std::find(takenPositions.begin(), takenPositions.end(), *it) == takenPositions.end()) {

			// Set the agent's position 
			agent->setX((*it).first);
			agent->setY((*it).second);

			break;
		}
	}
}

/// Returns the list of neighbors within dist of the point x/y. This
/// can be the position of an agent, but it is not limited to this.
/// \date    2012-01-29
/// \return  The list of neighbors
/// \param   x the x coordinate
/// \param   y the y coordinate
/// \param   dist the distance around x/y that will be searched for agents (search field is a square in the current implementation)
set<const Ped::Tagent*> Ped::Model::getNeighbors(int x, int y, int dist) const {

	// create the output list
	// ( It would be better to include only the agents close by, but this programmer is lazy.)	
	return set<const Ped::Tagent*>(agents.begin(), agents.end());
}

void Ped::Model::cleanup() {
	finished.store(true);
	
	while (barrier_1.get_waiting() != nr_threads)
		std::this_thread::sleep_for(nano_secs);

	barrier_1.release();

	while (barrier_2.get_waiting() != nr_threads)
		std::this_thread::sleep_for(nano_secs);

	barrier_2.release();
	
	for (auto &i : threads) i.join();
	// Nothing to do here right now. 
}

Ped::Model::~Model()
{
	std::for_each(agents.begin(), agents.end(), [](Ped::Tagent *agent){delete agent;});
	std::for_each(destinations.begin(), destinations.end(), [](Ped::Twaypoint *destination){delete destination; });
}
