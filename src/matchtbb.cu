/*
Copyright 2011, Bas Fagginger Auer.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/
#include <iostream>
#include <exception>
#include <string>
#include <list>
#include <map>
#include <queue>
#include <algorithm>

#include <tbb/tbb.h>

#include <cassert>

#include "matchtbb.h"

using namespace tbb;
using namespace mtc;
using namespace std;

//FIXME: This is not a very OO way to implement a global stopping flag.
bool keepMatchingTBB;

GraphMatchingTBB::GraphMatchingTBB(const Graph &_graph, const unsigned int &_selectBarrier) :
	GraphMatching(_graph),
	selectBarrier(_selectBarrier)
{

}

GraphMatchingTBB::~GraphMatchingTBB()
{
	
}

#define LEFTROTATE(a, b) (((a) << (b)) | ((a) >> (32 - (b))))

//==== General matching TBB classes ====
class SelectVertices
{
	public:
		SelectVertices(int *, const int &, const uint &);
		~SelectVertices();
		
		void setSeed(const uint &);
		void operator () (const blocked_range<int> &) const;
	
	private:
		int *match;
		const int nrVertices;
		uint random;
		const uint selectBarrier;
		
		static const uint MD5K[64];
		static const uint MD5R[64];
};

class MatchVertices
{
	public:
		MatchVertices(int *, const int *, const int &);
		~MatchVertices();
		
		void operator () (const blocked_range<int> &) const;
	
	private:
		int *match;
		const int *requests;
		const int nrVertices;
};

//Nothing-up-my-sleeve working constants from SHA-256.
const uint SelectVertices::MD5K[64] = {0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
				0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
				0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
				0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
				0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
				0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
				0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
				0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};

		//Rotations from MD5.
const uint SelectVertices::MD5R[64] = {7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22,
				5,  9, 14, 20, 5,  9, 14, 20, 5,  9, 14, 20, 5,  9, 14, 20,
				4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23,
				6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21};

SelectVertices::SelectVertices(int *_match, const int &_nrVertices, const uint &_selectBarrier) :
	match(_match),
	nrVertices(_nrVertices),
	random(0),
	selectBarrier(_selectBarrier)
{

};

SelectVertices::~SelectVertices()
{

};

void SelectVertices::setSeed(const uint &_random)
{
	random = _random;
};

void SelectVertices::operator () (const blocked_range<int> &r) const
{
	//Apply selection procedure to each pi-value in parallel.
	for (int i = r.begin(); i != r.end(); ++i)
	{
		//This code should be the same as in matchgpu.cu!
		if (match[i] >= 2) continue;
		
		//There are still vertices to be matched.
		keepMatchingTBB = true;
		
		//Start hashing.
		uint h0 = 0x67452301, h1 = 0xefcdab89, h2 = 0x98badcfe, h3 = 0x10325476;
		uint a = h0, b = h1, c = h2, d = h3, e, f, g = i;

		for (int j = 0; j < 16; ++j)
		{
			f = (b & c) | ((~b) & d);

			e = d;
			d = c;
			c = b;
			b += LEFTROTATE(a + f + MD5K[j] + g, MD5R[j]);
			a = e;

			h0 += a;
			h1 += b;
			h2 += c;
			h3 += d;

			g *= random;
		}
		
		match[i] = ((h0 + h1 + h2 + h3) < selectBarrier ? 0 : 1);
	}
};

MatchVertices::MatchVertices(int *_match, const int *_requests, const int &_nrVertices) :
	match(_match),
	requests(_requests),
	nrVertices(_nrVertices)
{

};

MatchVertices::~MatchVertices()
{

};

void MatchVertices::operator () (const blocked_range<int> &s) const
{
	//Match all compatible requests in parallel.
	for (int i = s.begin(); i != s.end(); ++i)
	{
		const int r = requests[i];

		//Only unmatched vertices make requests.
		if (r == nrVertices + 1)
		{
			//This is vertex without any available neighbours, discard it.
			match[i] = 2;
		}
		else if (r < nrVertices)
		{
			//This vertex has made a valid request.
			if (requests[r] == i)
			{
				//Match the vertices if the request was mutual.
				match[i] = 4 + min(i, r);
			}
		}
	}
};

//==== Random greedy matching TBB classes ====
class MakeRandomRequests
{
	public:
		MakeRandomRequests(int *, const int *, const int &, const int2 *, const int *);
		~MakeRandomRequests();
		
		void operator () (const blocked_range<int> &) const;
	
	private:
		int *requests;
		const int *match;
		const int nrVertices;
		const int2 *neighbourRanges;
		const int *neighbours;
};

class MakeRandomResponses
{
	public:
		MakeRandomResponses(int *, const int *, const int &, const int2 *, const int *);
		~MakeRandomResponses();
		
		void operator () (const blocked_range<int> &) const;
	
	private:
		int *requests;
		const int *match;
		const int nrVertices;
		const int2 *neighbourRanges;
		const int *neighbours;
};

MakeRandomRequests::MakeRandomRequests(int *_requests, const int *_match, const int &_nrVertices, const int2 *_neighbourRanges, const int *_neighbours) :
	requests(_requests),
	match(_match),
	nrVertices(_nrVertices),
	neighbourRanges(_neighbourRanges),
	neighbours(_neighbours)
{

};

MakeRandomRequests::~MakeRandomRequests()
{
	
};
		
void MakeRandomRequests::operator () (const blocked_range<int> &s) const
{
	//Make requests in parallel.
	for (int i = s.begin(); i != s.end(); ++i)
	{
		//Look at all blue vertices and let them make requests.
		if (match[i] == 0)
		{
			const int2 indices = neighbourRanges[i];
			int dead = 1;

			for (int j = indices.x; j < indices.y; ++j)
			{
				const int ni = neighbours[j];
				const int nm = match[ni];

				//Do we have an unmatched neighbour?
				if (nm < 4)
				{
					//Is this neighbour red?
					if (nm == 1)
					{
						//Propose to this neighbour.
						requests[i] = ni;
						dead = 2;
						break;
					}
					else
					{
						dead = 0;
					}
				}
			}

			if (dead != 2) requests[i] = nrVertices + dead;
		}
		else
		{
			//Clear request value.
			requests[i] = nrVertices;
		}
	}
};

MakeRandomResponses::MakeRandomResponses(int *_requests, const int *_match, const int &_nrVertices, const int2 *_neighbourRanges, const int *_neighbours) :
	requests(_requests),
	match(_match),
	nrVertices(_nrVertices),
	neighbourRanges(_neighbourRanges),
	neighbours(_neighbours)
{

};

MakeRandomResponses::~MakeRandomResponses()
{
	
};
		
void MakeRandomResponses::operator () (const blocked_range<int> &s) const
{
	//Make requests in parallel.
	for (int i = s.begin(); i != s.end(); ++i)
	{
		//Look at all red vertices.
		if (match[i] == 1)
		{
			const int2 indices = neighbourRanges[i];

			//Select first available proposer.
			for (int j = indices.x; j < indices.y; ++j)
			{
				const int ni = neighbours[j];

				//Only respond to blue neighbours.
				if (match[ni] == 0)
				{
					//Avoid data thrashing be only looking at the request value of blue neighbours.
					if (requests[ni] == i)
					{
						requests[i] = ni;
						break;
					}
				}
			}
		}
	}
};

//==== Weighted greedy matching TBB classes ====
class MakeWeightedRequests
{
	public:
		MakeWeightedRequests(int *, const int *, const int &, const int2 *, const int *, const float *);
		~MakeWeightedRequests();
		
		void operator () (const blocked_range<int> &) const;
	
	private:
		int *requests;
		const int *match;
		const int nrVertices;
		const int2 *neighbourRanges;
		const int *neighbours;
		const float *neighbourWeights;
};

class MakeWeightedResponses
{
	public:
		MakeWeightedResponses(int *, const int *, const int &, const int2 *, const int *, const float *);
		~MakeWeightedResponses();
		
		void operator () (const blocked_range<int> &) const;
	
	private:
		int *requests;
		const int *match;
		const int nrVertices;
		const int2 *neighbourRanges;
		const int *neighbours;
		const float *neighbourWeights;
};

MakeWeightedRequests::MakeWeightedRequests(int *_requests, const int *_match, const int &_nrVertices, const int2 *_neighbourRanges, const int *_neighbours, const float *_neighbourWeights) :
	requests(_requests),
	match(_match),
	nrVertices(_nrVertices),
	neighbourRanges(_neighbourRanges),
	neighbours(_neighbours),
	neighbourWeights(_neighbourWeights)
{

};

MakeWeightedRequests::~MakeWeightedRequests()
{
	
};
		
void MakeWeightedRequests::operator () (const blocked_range<int> &s) const
{
	//Make requests in parallel.
	for (int i = s.begin(); i != s.end(); ++i)
	{
		//Look at all blue vertices and let them make requests.
		if (match[i] == 0)
		{
			const int2 indices = neighbourRanges[i];
			float maxWeight = -1.0;
			int candidate = nrVertices;
			int dead = 1;

			for (int j = indices.x; j < indices.y; ++j)
			{
				//Only propose to red neighbours.
				const int ni = neighbours[j];
				const int nm = match[ni];

				//Do we have an unmatched neighbour?
				if (nm < 4)
				{
					//Is this neighbour red?
					if (nm == 1)
					{
						//Propose to the heaviest neighbour.
						const float nw = neighbourWeights[j];

						if (nw > maxWeight)
						{
							maxWeight = nw;
							candidate = ni;
						}
					}
					
					dead = 0;
				}
			}

			requests[i] = candidate + dead;
		}
		else
		{
			//Clear request value.
			requests[i] = nrVertices;
		}
	}
};

MakeWeightedResponses::MakeWeightedResponses(int *_requests, const int *_match, const int &_nrVertices, const int2 *_neighbourRanges, const int *_neighbours, const float *_neighbourWeights) :
	requests(_requests),
	match(_match),
	nrVertices(_nrVertices),
	neighbourRanges(_neighbourRanges),
	neighbours(_neighbours),
	neighbourWeights(_neighbourWeights)
{

};

MakeWeightedResponses::~MakeWeightedResponses()
{
	
};
		
void MakeWeightedResponses::operator () (const blocked_range<int> &s) const
{
	//Make requests in parallel.
	for (int i = s.begin(); i != s.end(); ++i)
	{
		//Look at all red vertices.
		const int2 indices = neighbourRanges[i];
		float maxWeight = -1;
		int candidate = nrVertices;

		//Select heaviest available proposer.
		for (int j = indices.x; j < indices.y; ++j)
		{
			const int ni = neighbours[j];

			//Only respond to blue neighbours.
			if (match[ni] == 0)
			{
				if (requests[ni] == i)
				{
					const float nw = neighbourWeights[j];

					if (nw > maxWeight)
					{
						maxWeight = nw;
						candidate = ni;
					}
				}
			}
		}

		if (candidate < nrVertices)
		{
			requests[i] = candidate;
		}
	}
};

GraphMatchingTBBRandom::GraphMatchingTBBRandom(const Graph &_graph, const unsigned int &_selectBarrier) :
	GraphMatchingTBB(_graph, _selectBarrier)
{

}

GraphMatchingTBBRandom::~GraphMatchingTBBRandom()
{
	
}

void GraphMatchingTBBRandom::performMatching(vector<int> &match, cudaEvent_t &t1, cudaEvent_t &t2, vector<int> & h, vector<int> & t, vector<int> & fll) const
{
	//This is a random greedy matching algorithm using Intel's Threading Building Blocks library.
	//Assumes that the order of the vertices has already been randomized.
	
	assert((int)match.size() == graph.nrVertices);
	
	//Clear matching.
	match.assign(graph.nrVertices, 0);
	
	//Create requests array.
	vector<int> requests(graph.nrVertices, 0);
	
	cudaEventRecord(t1, 0);
	cudaEventSynchronize(t1);
	
	//Initialise kernels.
	SelectVertices selector(&match[0], graph.nrVertices, selectBarrier);
	MakeRandomRequests requester(&requests[0], &match[0], graph.nrVertices, &graph.neighbourRanges[0], &graph.neighbours[0]);
	MakeRandomResponses responder(&requests[0], &match[0], graph.nrVertices, &graph.neighbourRanges[0], &graph.neighbours[0]);
	MatchVertices matcher(&match[0], &requests[0], graph.nrVertices);
	const blocked_range<int> range(0, graph.nrVertices);
	
	int count = 0;
	keepMatchingTBB = true;
	
	//for (int i = 0; i < NR_MATCH_ROUNDS; ++i)
	while (keepMatchingTBB && ++count < NR_MAX_MATCH_ROUNDS)
	{
		keepMatchingTBB = false;
		selector.setSeed(rand());
		
		parallel_for(range, selector);
		parallel_for(range, requester);
		parallel_for(range, responder);
		parallel_for(range, matcher);
		/*
		tbb::tick_count u0 = tick_count::now();
		
		selector.setSeed(rand());
		
		parallel_for(range, selector);
		tbb::tick_count u1 = tick_count::now();
		parallel_for(range, requester);
		tbb::tick_count u2 = tick_count::now();
		parallel_for(range, responder);
		tbb::tick_count u3 = tick_count::now();
		parallel_for(range, matcher);
		tbb::tick_count u4 = tick_count::now();
		
		cout << "TIMINGS:\t" << scientific << (u1 - u0).seconds() << "\t" << (u2 - u1).seconds() << "\t" << (u3 - u2).seconds() << "\t" << (u4 - u3).seconds() << endl;
		*/
	}
	
	cudaEventRecord(t2, 0);
	cudaEventSynchronize(t2);
}

GraphMatchingTBBWeighted::GraphMatchingTBBWeighted(const Graph &_graph, const unsigned int &_selectBarrier) :
	GraphMatchingTBB(_graph, _selectBarrier)
{

}

GraphMatchingTBBWeighted::~GraphMatchingTBBWeighted()
{
	
}

void GraphMatchingTBBWeighted::performMatching(vector<int> &match, cudaEvent_t &t1, cudaEvent_t &t2, vector<int> & h, vector<int> & t, vector<int> & fll) const
{
	//This is a weighted greedy matching algorithm using Intel's Threading Building Blocks library.
	//Assumes that the order of the vertices has already been randomized.
	
	assert((int)match.size() == graph.nrVertices);
	
	//Clear matching.
	match.assign(graph.nrVertices, 0);
	
	//Create requests array.
	vector<int> requests(graph.nrVertices, 0);
	
	cudaEventRecord(t1, 0);
	cudaEventSynchronize(t1);
	
	//Initialise kernels.
	SelectVertices selector(&match[0], graph.nrVertices, selectBarrier);
	MakeWeightedRequests requester(&requests[0], &match[0], graph.nrVertices, &graph.neighbourRanges[0], &graph.neighbours[0], &graph.neighbourWeights[0]);
	MakeWeightedResponses responder(&requests[0], &match[0], graph.nrVertices, &graph.neighbourRanges[0], &graph.neighbours[0], &graph.neighbourWeights[0]);
	MatchVertices matcher(&match[0], &requests[0], graph.nrVertices);
	const blocked_range<int> range(0, graph.nrVertices);
	
	int count = 0;
	keepMatchingTBB = true;
	
	//for (int i = 0; i < NR_MATCH_ROUNDS; ++i)
	while (keepMatchingTBB && ++count < NR_MAX_MATCH_ROUNDS)
	{
		keepMatchingTBB = false;
		selector.setSeed(rand());
		
		parallel_for(range, selector);
		parallel_for(range, requester);
		parallel_for(range, responder);
		parallel_for(range, matcher);
		/*
		tbb::tick_count u0 = tick_count::now();
		
		selector.setSeed(rand());
		
		parallel_for(range, selector);
		tbb::tick_count u1 = tick_count::now();
		parallel_for(range, requester);
		tbb::tick_count u2 = tick_count::now();
		parallel_for(range, responder);
		tbb::tick_count u3 = tick_count::now();
		parallel_for(range, matcher);
		tbb::tick_count u4 = tick_count::now();
		
		cout << "TIMINGS:\t" << scientific << (u1 - u0).seconds() << "\t" << (u2 - u1).seconds() << "\t" << (u3 - u2).seconds() << "\t" << (u4 - u3).seconds() << endl;
		*/
	}
	
	cudaEventRecord(t2, 0);
	cudaEventSynchronize(t2);
}

