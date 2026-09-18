#pragma once
//SMARTCACHE RUNG POLICY (DP-370). One function decides whether every ladder rung exists - the
//prefill save's promotion, a prefill seed, and what eviction frees - so the three can no longer
//disagree. It has no llama dependencies so it can be tested on its own:
//tests/smartcache_rung_policy_test.cpp checks it against brute force.
//
//Input: candidate rungs sorted by depth. The first node is the origin (depth 0, no bytes) and
//the last is where the context ends. Both are mandatory, and so is anything that cannot be
//given up (the head checkpoint mid-prefill, a slot a load is about to read).
//
//A gap from rung a up to rung b costs, at worst, w(b) * (T(b) - T(a)): an edit just below b
//reprocesses everything from a. T is cumulative prefill cost, so a deep gap is priced by how
//slowly deep tokens prefill, not by its token count. w is how much an edit landing there matters.
//
//The chosen set:
//  1. minimises N = its largest gap cost, subject to its bytes fitting the budget;
//  2. among sets with every gap <= N, minimises sum over gaps of w(b)*(b-a)*(T(b)-T(a)), the
//     expected reprocessing for an edit landing uniformly inside a gap. Without this step a rung
//     that splits anything but the worst gap never changes N, and a ladder with free budget would
//     never grow. (Summing w*(T(b)-T(a)) instead is useless: over consecutive gaps it telescopes
//     to a constant.)
//  3. ties go to fewer bytes.
//
//Bytes are rounded UP to 1/budget_units of the budget, so a chosen set can never exceed the
//budget; it may leave up to one unit per node unspent.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

namespace smartcache_policy {

struct Node
{
    int depth = 0;
    double bytes = 0.0;
    bool mandatory = false;
    int tag = -1; //the caller's handle: a slot index, or a negative marker for a virtual rung
};

struct Plan
{
    bool feasible = false;  //false: even the mandatory nodes do not fit, and only they are kept
    double worst = 0.0;     //N - the largest gap cost in the chosen set
    double expected = 0.0;  //sum of w*(b-a)*(T(b)-T(a)) over the chosen set's gaps
    double bytes = 0.0;
    std::vector<char> keep; //parallel to the input nodes
};

static const int budget_units = 4096;

template<typename TFn, typename WFn>
Plan plan(const std::vector<Node> &nodes, double budget, TFn T, WFn w)
{
    Plan out;
    const int m = (int)nodes.size();
    out.keep.assign(m, 0);
    if(m == 0)
    {
        return out;
    }

    std::vector<double> tv(m), wv(m);
    for(int i=0;i<m;++i)
    {
        tv[i] = T(nodes[i].depth);
        wv[i] = w(nodes[i].depth);
    }
    auto worst_of = [&](int a, int b) { return wv[b]*(tv[b]-tv[a]); };
    auto expected_of = [&](int a, int b) { return wv[b]*(double)(nodes[b].depth-nodes[a].depth)*(tv[b]-tv[a]); };
    auto finish = [&]() {
        int prev = -1;
        for(int i=0;i<m;++i)
        {
            if(!out.keep[i])
            {
                continue;
            }
            out.bytes += nodes[i].bytes;
            if(prev >= 0)
            {
                out.worst = std::max(out.worst, worst_of(prev, i));
                out.expected += expected_of(prev, i);
            }
            prev = i;
        }
        return out;
    };

    //a gap may not jump over a mandatory node: the nearest mandatory node below each index
    std::vector<int> prevmand(m, 0);
    int lastm = 0;
    for(int j=0;j<m;++j)
    {
        prevmand[j] = lastm;
        if(nodes[j].mandatory)
        {
            lastm = j;
        }
    }

    if(budget <= 0.0) //no budget to fit: everything is kept
    {
        std::fill(out.keep.begin(), out.keep.end(), 1);
        out.feasible = true;
        return finish();
    }

    const double unit = budget / (double)budget_units;
    std::vector<int> units(m);
    int mandunits = 0;
    for(int i=0;i<m;++i)
    {
        const double u = std::ceil(nodes[i].bytes / unit - 1e-9);
        units[i] = (u <= 0.0 ? 0 : (u > (double)budget_units ? budget_units + 1 : (int)u));
        if(nodes[i].mandatory || i==0 || i==m-1)
        {
            mandunits += units[i];
        }
    }
    if(mandunits > budget_units)
    {
        for(int i=0;i<m;++i)
        {
            out.keep[i] = (nodes[i].mandatory || i==0 || i==m-1) ? 1 : 0;
        }
        out.feasible = false;
        return finish();
    }

    //1. the smallest N whose cheapest covering set fits
    const int INF = std::numeric_limits<int>::max() / 2;
    std::vector<double> candidates;
    for(int j=1;j<m;++j)
    {
        for(int i=prevmand[j];i<j;++i)
        {
            candidates.push_back(worst_of(i, j));
        }
    }
    std::sort(candidates.begin(), candidates.end());
    candidates.erase(std::unique(candidates.begin(), candidates.end()), candidates.end());
    auto min_units = [&](double N) {
        std::vector<int> dp(m, INF);
        dp[0] = units[0];
        for(int j=1;j<m;++j)
        {
            for(int i=prevmand[j];i<j;++i)
            {
                if(dp[i] < INF && worst_of(i, j) <= N)
                {
                    dp[j] = std::min(dp[j], dp[i] + units[j]);
                }
            }
        }
        return dp[m-1];
    };
    int lo = 0, hi = (int)candidates.size() - 1; //hi always fits: it admits the mandatory-only chain
    if(hi < 0)
    {
        std::fill(out.keep.begin(), out.keep.end(), 1); //a single node
        out.feasible = true;
        return finish();
    }
    while(lo < hi)
    {
        const int mid = (lo + hi) / 2;
        if(min_units(candidates[mid]) <= budget_units)
        {
            hi = mid;
        }
        else
        {
            lo = mid + 1;
        }
    }
    const double nstar = candidates[lo];

    //2. least expected reprocessing among sets with every gap <= N that fit, 3. then fewest bytes
    const int W = budget_units + 1;
    const double DINF = std::numeric_limits<double>::infinity();
    std::vector<double> best((size_t)m * W, DINF);
    std::vector<int> from((size_t)m * W, -1);
    best[units[0]] = 0.0;
    for(int j=1;j<m;++j)
    {
        const int uj = units[j];
        if(uj > budget_units)
        {
            continue;
        }
        for(int i=prevmand[j];i<j;++i)
        {
            if(worst_of(i, j) > nstar)
            {
                continue;
            }
            const double c = expected_of(i, j);
            const size_t src = (size_t)i * W, dst = (size_t)j * W;
            for(int u=0;u+uj<=budget_units;++u)
            {
                const double e = best[src + u];
                if(e == DINF)
                {
                    continue;
                }
                if(e + c < best[dst + u + uj])
                {
                    best[dst + u + uj] = e + c;
                    from[dst + u + uj] = i;
                }
            }
        }
    }
    int bu = -1;
    for(int u=0;u<=budget_units;++u)
    {
        const double e = best[(size_t)(m-1) * W + u];
        if(e != DINF && (bu < 0 || e < best[(size_t)(m-1) * W + bu]))
        {
            bu = u;
        }
    }
    if(bu < 0) //unreachable: step 1 proved a set exists
    {
        for(int i=0;i<m;++i)
        {
            out.keep[i] = (nodes[i].mandatory || i==0 || i==m-1) ? 1 : 0;
        }
        return finish();
    }
    for(int j=m-1, u=bu; j>=0;)
    {
        out.keep[j] = 1;
        if(j == 0)
        {
            break;
        }
        const int i = from[(size_t)j * W + u];
        u -= units[j];
        j = i;
    }
    out.feasible = true;
    return finish();
}

} // namespace smartcache_policy
