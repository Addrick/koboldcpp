//Standalone check of otherarch/smartcache_rung_policy.h (DP-370): random instances against brute
//force over every subset, plus the behaviours the ladder relies on. No llama, no server.
//  g++ -O2 -std=c++17 -Iotherarch tests/smartcache_rung_policy_test.cpp -o srp_test && ./srp_test
//  cl /nologo /EHsc /O2 /std:c++17 /Iotherarch tests\smartcache_rung_policy_test.cpp && smartcache_rung_policy_test.exe
#include "smartcache_rung_policy.h"
#include <cmath>
#include <cstdio>
#include <functional>
#include <random>

using namespace smartcache_policy;

static int fails = 0;
static int checks = 0;
#define CHECK(cond, ...) do { ++checks; if(!(cond)) { ++fails; printf("FAIL line %d: ", __LINE__); printf(__VA_ARGS__); printf("\n"); } } while(0)

using Fn = std::function<double(int)>;

static bool is_fixed(const std::vector<Node> &n, int i)
{
    return i==0 || i==(int)n.size()-1 || n[i].mandatory;
}

static int units_of(double bytes, double budget)
{
    const double u = std::ceil(bytes / (budget / budget_units) - 1e-9);
    return u <= 0.0 ? 0 : (int)u;
}

struct Score
{
    bool found = false;
    double worst = 0.0, expected = 0.0;
    int units = 0;
};

static Score score_of(const std::vector<Node> &n, const std::vector<char> &keep, double budget, const Fn &T, const Fn &w)
{
    Score s;
    s.found = true;
    int prev = -1;
    for(int i=0;i<(int)n.size();++i)
    {
        if(!keep[i])
        {
            continue;
        }
        s.units += units_of(n[i].bytes, budget);
        if(prev >= 0)
        {
            const double dt = T(n[i].depth) - T(n[prev].depth);
            s.worst = std::max(s.worst, w(n[i].depth) * dt);
            s.expected += w(n[i].depth) * (double)(n[i].depth - n[prev].depth) * dt;
        }
        prev = i;
    }
    return s;
}

static bool close(double a, double b)
{
    return std::fabs(a - b) <= 1e-9 * std::max(1.0, std::max(std::fabs(a), std::fabs(b)));
}

static Score brute(const std::vector<Node> &n, double budget, const Fn &T, const Fn &w)
{
    std::vector<int> free_idx;
    for(int i=0;i<(int)n.size();++i)
    {
        if(!is_fixed(n, i))
        {
            free_idx.push_back(i);
        }
    }
    Score best;
    for(long mask=0; mask < (1L << free_idx.size()); ++mask)
    {
        std::vector<char> keep(n.size(), 0);
        for(int i=0;i<(int)n.size();++i)
        {
            keep[i] = is_fixed(n, i) ? 1 : 0;
        }
        for(size_t k=0;k<free_idx.size();++k)
        {
            if(mask & (1L << k))
            {
                keep[free_idx[k]] = 1;
            }
        }
        Score s = score_of(n, keep, budget, T, w);
        if(s.units > budget_units)
        {
            continue;
        }
        const bool better = !best.found
            || (s.worst < best.worst && !close(s.worst, best.worst))
            || (close(s.worst, best.worst) && s.expected < best.expected && !close(s.expected, best.expected))
            || (close(s.worst, best.worst) && close(s.expected, best.expected) && s.units < best.units);
        if(better)
        {
            best = s;
        }
    }
    return best;
}

static std::vector<Node> ladder(const std::vector<int> &depths, double fixed, double rate)
{
    std::vector<Node> n;
    n.push_back({0, 0.0, true, -1});
    for(size_t i=0;i<depths.size();++i)
    {
        n.push_back({depths[i], fixed + rate * depths[i], false, (int)i});
    }
    n.back().mandatory = true; //the head
    return n;
}

static int kept_in(const std::vector<Node> &n, const Plan &p, int lo, int hi) //depth in (lo, hi]
{
    int c = 0;
    for(size_t i=1;i+1<n.size();++i)
    {
        c += (p.keep[i] && n[i].depth > lo && n[i].depth <= hi) ? 1 : 0;
    }
    return c;
}

int main()
{
    const Fn tokens = [](int d) { return (double)d; };
    const Fn flat = [](int) { return 1.0; };

    //--- slack: a budget that holds everything keeps everything
    {
        auto n = ladder({4096, 8192, 12288, 16384, 20480}, 100.0, 0.03);
        double total = 0.0;
        for(auto &x : n) total += x.bytes;
        Plan p = plan(n, total * 1.5, tokens, flat);
        int kept = 0;
        for(char k : p.keep) kept += k;
        CHECK(p.feasible && kept == (int)n.size(), "slack kept %d of %zu", kept, n.size());
    }

    //--- the head alone does not fit: infeasible, mandatory only
    {
        auto n = ladder({4096, 8192, 100000}, 100.0, 0.03);
        Plan p = plan(n, 1000.0, tokens, flat);
        CHECK(!p.feasible && p.keep[0] && p.keep.back() && !p.keep[1] && !p.keep[2], "infeasible head");
    }

    //--- a mandatory node in the middle is never skipped, even when it is expensive and useless
    {
        auto n = ladder({1000, 2000, 3000, 4000, 5000, 6000}, 10.0, 0.0);
        n[2].mandatory = true;
        n[2].bytes = 40.0;
        Plan p = plan(n, 75.0, tokens, flat);
        CHECK(p.feasible && p.keep[2], "mandatory middle node dropped");
    }

    //--- step weight: with room for half the rungs, the recent region stays dense and old history thins
    {
        std::vector<int> d;
        for(int x=2048; x<=65536; x+=2048) d.push_back(x);
        auto n = ladder(d, 50.0, 0.0);
        static const int edge = 49152;
        const Fn step = [](int b) { return b > edge ? 1.0 : 0.01; };
        Plan p = plan(n, 50.0 * 16, tokens, step);
        const int recent = kept_in(n, p, edge, 65535), recent_all = 7;
        const int old = kept_in(n, p, 0, edge), old_all = 24;
        CHECK(p.feasible && recent == recent_all, "recent region kept %d of %d", recent, recent_all);
        CHECK((double)old / old_all < (double)recent / recent_all, "old %d/%d not thinner than recent %d/%d", old, old_all, recent, recent_all);
    }

    //--- uniform weight and cost: the kept rungs spread evenly instead of clustering
    {
        std::vector<int> d;
        for(int x=1000; x<=40000; x+=1000) d.push_back(x);
        auto n = ladder(d, 10.0, 0.0);
        Plan p = plan(n, 10.0 * 5.01, tokens, flat); //head + 4 rungs; a hair over, since bytes round up to budget units
        CHECK(p.feasible && close(p.worst, 8000.0), "4 rungs over 40000 should give worst gap 8000, got %.0f", p.worst);
    }

    //--- time pricing: when deep tokens prefill slower, deep gaps come out narrower than shallow ones
    {
        std::vector<int> d;
        for(int x=1000; x<=60000; x+=1000) d.push_back(x);
        auto n = ladder(d, 10.0, 0.0);
        const Fn slow_deep = [](int x) { return (double)x + (double)x * (double)x / 20000.0; };
        Plan p = plan(n, 10.0 * 7, slow_deep, flat);
        int first = -1, last_gap = 0, prev = 0;
        for(size_t i=1;i<n.size();++i)
        {
            if(!p.keep[i]) continue;
            if(first < 0) first = n[i].depth - prev;
            last_gap = n[i].depth - prev;
            prev = n[i].depth;
        }
        CHECK(p.feasible && last_gap < first, "deep gap %d not narrower than shallow %d", last_gap, first);
    }

    //--- a hole beats thinning: one more rung's worth of budget goes into the widest gap
    {
        auto n = ladder({2000, 4000, 6000, 8000, 30000, 50000}, 10.0, 0.0);
        n[6].mandatory = false;
        n.push_back({52000, 10.0, true, 99}); //head
        //rungs at 2000..8000 and 50000; 30000 sits in the 8000..50000 hole; room for the head + 3
        Plan p = plan(n, 10.0 * 4, tokens, flat);
        CHECK(p.feasible && p.keep[5], "the rung in the widest hole was not kept");
    }

    //--- random instances against brute force
    std::mt19937 rng(370);
    int instances = 0;
    for(int it=0; it<4000; ++it)
    {
        std::uniform_int_distribution<int> count(0, 12);
        const int k = count(rng) + 1;
        std::vector<int> d;
        int x = 0;
        for(int i=0;i<k;++i)
        {
            x += std::uniform_int_distribution<int>(1, 9000)(rng);
            d.push_back(x);
        }
        const double fixed = std::uniform_real_distribution<double>(0.0, 300.0)(rng);
        const double rate = std::uniform_real_distribution<double>(0.0, 0.05)(rng);
        auto n = ladder(d, fixed, rate);
        for(size_t i=1;i+1<n.size();++i)
        {
            n[i].mandatory = (std::uniform_int_distribution<int>(0, 9)(rng) == 0);
        }
        double total = 0.0;
        for(auto &y : n) total += y.bytes;
        const double budget = std::uniform_real_distribution<double>(0.05, 1.2)(rng) * (total + 1.0);

        const int shape = it % 3;
        const double a = std::uniform_real_distribution<double>(0.0, 1e-4)(rng);
        Fn T = shape == 0 ? tokens
             : shape == 1 ? Fn([a](int v) { return (double)v + a * (double)v * (double)v; })
                          : Fn([](int v) { return std::sqrt((double)v) * 100.0 + (double)v * 0.1; });
        const int edge = std::uniform_int_distribution<int>(0, x)(rng);
        const double far = std::uniform_real_distribution<double>(0.0, 1.0)(rng);
        Fn w = (it % 2) ? flat : Fn([edge, far](int b) { return b > edge ? 1.0 : far; });

        Plan p = plan(n, budget, T, w);
        Score b = brute(n, budget, T, w);
        Score got = score_of(n, p.keep, budget, T, w);
        ++instances;
        if(!b.found)
        {
            CHECK(!p.feasible, "it %d: brute found nothing, plan claims feasible", it);
            continue;
        }
        CHECK(p.feasible, "it %d: plan infeasible, brute found worst %.3f", it, b.worst);
        CHECK(got.units <= budget_units, "it %d: plan over budget, %d units", it, got.units);
        for(size_t i=0;i<n.size();++i)
        {
            if(is_fixed(n, (int)i))
            {
                CHECK(p.keep[i], "it %d: fixed node %zu dropped", it, i);
            }
        }
        CHECK(close(got.worst, b.worst), "it %d: worst %.6f vs brute %.6f", it, got.worst, b.worst);
        CHECK(close(got.expected, b.expected), "it %d: expected %.6f vs brute %.6f", it, got.expected, b.expected);
        if(close(got.expected, b.expected))
        {
            CHECK(got.units == b.units, "it %d: units %d vs brute %d", it, got.units, b.units);
        }
    }

    printf("%d checks, %d instances, %d failures\n", checks, instances, fails);
    return fails ? 1 : 0;
}
