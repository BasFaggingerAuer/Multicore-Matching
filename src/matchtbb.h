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
#ifndef MATCH_MATCH_TBB_H
#define MATCH_MATCH_TBB_H

#include <vector>

#include "graph.h"
#include "matchcpu.h"

namespace mtc
{

class GraphMatchingTBB : public GraphMatching
{
	public:
		GraphMatchingTBB(const Graph &, const unsigned int &);
		~GraphMatchingTBB();
		
		virtual void performMatching(std::vector<int> &, cudaEvent_t &, cudaEvent_t &, std::vector<int> & h, std::vector<int> & t, std::vector<int> & fll) const = 0;
		
	protected:
		const uint selectBarrier;
};


class GraphMatchingTBBRandom : public GraphMatchingTBB
{
	public:
		GraphMatchingTBBRandom(const Graph &, const unsigned int &);
		~GraphMatchingTBBRandom();
		
		void performMatching(std::vector<int> &, cudaEvent_t &, cudaEvent_t &, std::vector<int> & h, std::vector<int> & t, std::vector<int> & fll) const;
};

class GraphMatchingTBBWeighted : public GraphMatchingTBB
{
	public:
		GraphMatchingTBBWeighted(const Graph &, const unsigned int &);
		~GraphMatchingTBBWeighted();
		
		void performMatching(std::vector<int> &, cudaEvent_t &, cudaEvent_t &, std::vector<int> & h, std::vector<int> & t, std::vector<int> & fll) const;
};

};

#endif
