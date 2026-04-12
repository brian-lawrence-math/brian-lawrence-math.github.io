---
layout: post
title: Speed-running Integer Factorization with AI
date:   2026-04-12 09:38:08 -0700
categories: jekyll update
---

The [quadratic sieve][quad-sieve] is a fast integer factorization algorithm:
if a positive integer $N$ is the product of two large prime numbers $$N = pq$$,
then quadratic sieve will find the two factors reasonably quickly.
"Reasonably" is the key word here: if $$N$$ is an $$n$$-bit integer
(so $$N \approx 2^n$$), then the runtime of quadratic sieve is something like $$2^{\sqrt{n}}$$.
This is much faster than trial division (runtime $$\sqrt{N} \approx 2^{n/2}$$)
but still not even close to a polynomial-time algorithm.

// delete this?  too much theory

Integer factorization is relevant because, if you can factor large integers,
you can break the [RSA cryptosystem][RSA].
In fact, it's precisely because of quadratic sieve and [algorithms like it][gnfs]
that many systems have switched over from RSA to [elliptic curve cryptography][ECC]
in recent years.

Anyway, I'm going to implement quadratic sieve on an [NVidia RTX 3050][rtx],
with some help from OpenAI's Codex.
I want to see what size of integer I can factor in a reasonable amount of time
(let's say one minute or so).
This will be a good excuse for me to explore both
- the challenges of adapting an interesting algorithm for parallelization, and
- what an AI agent can do.

This is a work in progress: I've gone through a couple of iterations
with Codex, and I've already factored some pretty big numbers,
but I'm still experimenting to see just how much I can get the AI to do.

## Summary: what coding agents can and can't do

Before I get to the juicy details I want to lead with what I've learned from working with Codex.

Codex is very, very good at basic coding tasks.
It never forgets semicolons.  Its code always compiles.
It knows how to organize a project, 
how to start by building something simple that works and make iterative improvements from there.

It's very good at reading and writing code. It quickly spots the sort of edge cases and off-by-one errors that used to cause me so much grief.

Codex also knows, in broad strokes, how the quadratic sieve algorithm is supposed to work.
"First you choose a factor base, then you find relations, then you solve a binary matrix."

So working with Codex is a tremendous productivity boost.

But Codex is reluctant to do any sort of experimentation or analysis, beyond "write code and see if it works".
Throughout the project, I had to give it two kinds of prompts:
- Requests for empirical data:
Even though I told Codex I was interested in optimizing performance,
the agent was reluctant to run experiments and gather data.
For example:
  - The agent gave up on a small factorization task because it only found half the necessary number of relations.
  But I told Codex the target was 60 seconds, and the program only ran for about 1 second.
  The solution -- double or triple the size of the search space -- would have been obvious
  if Codex had measured the runtime.
  - On a longer-running program, the agent put a bunch of effort into optimizing a trivially small code path
  (the is_squarefree() function).
  This function took one millisecond out of a total of about 4 seconds of execution.
  Again, a misdirected effort that could have been prevented by some simple timing.
  In fact, it's even worse: (1) Codex already had timing data in context, it just had to look, and 
  (2) even without experimental data, it should have been clear that is_squarefree() just isn't the bottleneck here.

- Theoretical analysis (asymptotic estimates, etc.).
  I found that Codex could do this sort of thing pretty well if I held its hand,
  but by default Codex would rather code than think.
  - The code was struggling to find enough relations, so Codex suggested decreasing the factor base bound $$B$$.
  In fact the solution is precisely the opposite.
  - I got a huge performance boost by figuring out the best search space to search for relations.
  Codex can help with this but only if I ask a long series of pointed leading questions.

In summary: Codex is a terrific coder and a great productivity boost.
If I want to get more out of it, the next challenge is to get it to divide its efforts
among implementation, empirical data-gathering, and theoretical analysis.

## The algorithm

OK, with that summary out of the way, let's get back to a brief overview of quadratic sieve.
I want to give you a sense of
- the different parameter choices and how they affect performance, and
- how amenable things are to parallelization on GPU.



// link to Python implementation

Just to fix ideas, imagine that $N = pq$ is a 100-200 bit integer (30-70 digits),
and the primes $p$ and $q$ are about the same size.
With these parameters, we'd like things to run in a few seconds on a GPU.

The algorithm has two steps (not counting pre- and post-processing, which should be "fast").

- Preprocessing: Choose a factor base bound $B$ (for us, maybe $1000$, $10000$, $100000$) and a search plan for the next step.
- Search for "relations".  A relation is a pair $(k, a)$ such that
\\[
\text{all the prime factors of } a^2 - kN    \text{ are less than } B.
\\]
In other words:
\\[ a^2 \equiv \text{product of small primes (mod } N\text{).} \\]
The number of relations you need to find is a few more than the number of primes less than $B$
(which is about $B / \log B$).
- Solve a linear system: Combine some of the relations to get a handful of perfect-square relations of the form
\\[ a^2 \equiv b^2 \text{(mod } N \text{).} \\]
This step amounts to solving a large binary matrix.
- Postprocessing: For each perfect-square relation, you know that 
\\[ a^2 - b^2 = (a + b) (a - b) \\]
is a multiple of $N$.  
Compute
\\[ \operatorname{gcd}(a+b, N) \\]
using the Euclidean algorithm.
There is a 50% chance that this gcd will be a nontrivial factor of $N$ (either $p$ or $q$), 
in which case you are done.
If not, repeat with more perfect-square relations until you find a factor.

### Asymptotics

In choosing $B$, there is a tradeoff between the search for relations and solving the linear system.
If $B$ is too small, relations will be hard to find (it's a very rare number that is only divisible by 2, 3 or 5!).
But the larger $B$ is, the larger the matrix that needs to be solved in the solving step.

Here are some impressionistic asymptotics: I'm ignoring some important log factors and constants
so I can paint a clear picture in your head.
The goal is not to get precise runtime estimates (those come from experiment!)
but to give you a mental model of the tradeoffs.

Suppose $B$ is a $b$-bit integer, and suppose $N \approx B^k$.  In other words:
\\[  B \approx 2^b \\]
\\[  N \approx 2^n \\]
\\[  n = bk.  \\]

Then the search for relations will take something like $2^{2k}$ time -- the smaller $B$ is compared to $N$, the rarer these relations are.

Solving the linear system will take something like $2^{3b}$ time -- you're solving a matrix of size (approximately) $B$,
and solving a matrix (at least by the naive algorithm) takes cubic time.

Now you can see where the $2^{\sqrt{n}}$ asymptotic comes from.  Suppose you have $2^t$ total time budget.
You want to divide it close to evenly between search and solve;
let's say you allocate $2^t$ for each step.
(Oops!  Did I just double your budget?
Don't worry, this is already a sloppy estimate, one more factor of 2 is no big deal.)
This means you want
\\[ k = t/2 \\]
\\[ b = t/3 \\]
so you can factor a number of up to
\\[ n = bk = t^2 / 6 \\]
bits.

In practice, this means:
- The algorithm has two phases, search and solve, which can be run, timed, and optimized independently.
- Increasing $B$ makes the search phase faster and the solve phase slower.
- The solve phase runtime depends only on the size of $B$, not on $N$.  In practice, $B$ in the tens of thousands might take a few seconds on a CPU.

### More analysis: the search space.

Like I said before, we want 
\\[ a^2 - k N \\]
to have a factorization into lots of small factors.
Of course, this is most likely to happen if $a^2 - kN$ is small!
So the natural thing to do is to pick some smallish integer $k$, and then search for $a$ in some interval around $\sqrt{kN}$.
In other words, we'll look at
\\[ a = x + \left \lfloor \sqrt{kN} \right \rfloor \\]
where $k$ is a small positive integer, and $x$ is a small integer (positive or negative).

What sorts of $k$ and $x$ should we search over?
I think it's pretty clear that we should figure out which $k$ and $x$ will make
\\[ a^2 - k N \\]
be the smallest, and target those first.

With a little bit of algebra, you can find that
\\[ a^2 - k N \approx 2 x \sqrt{kN} \\]
when $x$ is small.  So the obvious strategy is to pick some bound $M$ and search all pairs $k$ and $x$
for which
\\[ -M < 2x \sqrt{kN} < M. \\]

At least, I think this is obvious.  Codex doesn't.
Codex wants to pick a single value of $k$ and then search over $x$ in an ever-growing interval.
I wanted to see if I could get Codex to do the analysis my way;
after all, the calculations I just did are well within its abilities!
It took a surprising amount of coaxing, but in the end we got a 5-10x speedup on the search step.

OK, enough of theory, let's get down to the metal.

## Parallelizing the search, and an interesting tradeoff

Both of the two big steps (search and solve) could potentially benefit from parallelization,
but search is the natural first target.
After all, the search is what they call "embarrassingly parallel":
each pair $(k, x)$ either is a hit or it isn't.
On the other hand, imagine trying to solve a matrix -- think row reduction or Gauss-Jordan elimination --
in parallel.
([Wikipedia][row-reduce] has a nice animation.)
Maybe each thread can take responsibility for reducing a single row,
but each pivot row will need to be broadcast to the threads, one after another.
And that's not to mention some of the more complicated [optimizations][Wiedemann] 
that might be useful in our sparse setting.

So let's focus on doing the search in parallel.

At first glance, this seems easy enough.
Each thread takes responsibility for a single $a^2 - k N$.
The thread does a series of trial divisions (by 2, 3, 5, 7, ...),
until it reaches the bound $B$ -- at which point it either accepts or rejects.

But it turns out this simple approach introduces a whole bunch of duplicated computation!
To see why, think about a single prime factor, like 2027.
Your algorithm ends up testing each of these numbers individually for divisibility by 2027.
A much faster approach is to "sieve":
it turns out you can do just a single divisibility test and
mark off all the $a$ that make $a^2 - k N$ a multiple of 2027.
(As usual, [Wikipedia][sieve] has a terrific animation.
And notice that this particular optimization predates the invention of the GPU by quite some years.)

Now imagine you're searching, say, one million numbers.
You can either do one million trial divisions (in parallel, one number per thread),
or you can assign a single thread responsibility for finding all multiples of 2027
(parallelizing the work one prime per thread).
If you use the sieving approach, instead of one million trial divisions,
you do just one trial division, and then you "mark" about $1000000/2027 \approx 500$ 
values.  Should be an easy win!

But...

Computation is fast; memory is slow.
We've just gone from a pure-compute regime where each thread can
(more or less) do the full computation using only its own registers
to a random-access regime where each thread is making its own individual, unpredictable
writes to global memory.

On a GPU (at least most NVidia GPUs are like this), threads are organized in "warps" of 32.
All the threads in a warp execute the same instruction in lockstep,
on the same streaming multiprocessor, with access to the same low-latency shared memory.
(Actually, I'm simplifying things somewhat.
Threads are organized into "warps" of 32 and "blocks" of a larger size --
the programmer has some control over block size but 1024 is typical.
Threads in a warp execute at the same time; threads in a block share memory.)

Anyway, imagine we parallize the work "one prime per thread".
So our imaginary 2027 thread is sharing a warp (and memory etc.) with other threads responsible for different primes:
2029, 2039, 2053, and so forth.
At each iteration of the loop, each thread will compute the next multiple of its own prime,
and then issue a write request to... somewhere in this global array of 1 million items.
The write address won't be cached, because who could have predicted it?

Even worse, these 32 writes will widely spread across memory.
Writes to DRAM on a GPU come in 128-byte transactions, 
and if all 32 threads write to the same 128-byte block of memory, 
the writes can be "coalesced" into a single transaction.
But if each thread writes to multiples of a different prime, 
the chip will be forced to handle 32 different transactions of 128 bytes each.

And finally, to protect against race conditions (what if two different prime threads
try to write to the same spot at the same time?)
we'll need to use atomic operations -- introducing further inefficiency.

Empirically, switching from "one $x$ per thread" trial division to "one prime per thread" sieving
leads to a substantial (5-10x) slowdown in the search phase.

### A more efficient solution

(To follow)


{% highlight ruby %}
def print_hi(name)
  puts "Hi, #{name}"
end
print_hi('Tom')
#=> prints 'Hi, Tom' to STDOUT.
{% endhighlight %}


[quad-sieve]: https://en.wikipedia.org/wiki/Quadratic_sieve
[row-reduce]: https://en.wikipedia.org/wiki/Gaussian_elimination
[Wiedemann]: https://en.wikipedia.org/wiki/Block_Wiedemann_algorithm
[sieve]: https://en.wikipedia.org/wiki/Sieve_of_Eratosthenes
