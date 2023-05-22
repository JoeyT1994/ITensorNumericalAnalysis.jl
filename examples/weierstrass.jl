using ITensors
using ITensorNetworks
using NamedGraphs
using EllipsisNotation
using Graphs
using LaTeXStrings

using ITensorNetworks: delta_network, contract_inner, distance_to_leaf
using NamedGraphs: add_edges, random_bfs_tree, rename_vertices

using Plots
using Random

using Statistics

include("../src/QTT_utils.jl")

plot_font = "Computer Modern"
default(; fontfamily=plot_font, linewidth=1.5, framestyle=:box, label=nothing, grid=false)

"""ITensorNetwork representation of the weierstrass function (first nterms) over [0,1]: see https://mathworld.wolfram.com/WeierstrassFunction.html"""
function weierstrass_itn(
  s::IndsNetwork, vertex_map::Dict, a::Int64, nterms::Int64; svd_kwargs...
)
  ψ = sin_itn(s, vertex_map; a=0.0, k=Float64(pi))
  ψ[first(vertices(ψ))] *= (1.0 / pi)
  for n in 2:nterms
    omega = pi * (n^a)
    ψt = sin_itn(s, vertex_map; a=0.0, k=omega)
    ψt[first(vertices(ψt))] *= (1.0 / (omega))
    ψ = ψ + ψt
    if isa(ψ, TreeTensorNetwork)
      ψ = truncate(ψ; svd_kwargs...)
    end
  end

  return ψ
end

"""Evaluate Weierstrass(x) (first nterms) see: see https://mathworld.wolfram.com/WeierstrassFunction.html"""
function weierstrass(a::Int64, nterms::Int64, x::Float64)
  out = 0
  for n in 1:nterms
    omega = pi * (n^a)
    out += sin(omega * x) / omega
  end

  return out
end

function main()
  L = 31
  g = named_grid((L, 1))
  a = 2
  nterms = 200
  s = siteinds("S=1/2", g)

  #Define a map which determines a canonical ordering of the vertices of the network
  vertex_map = Dict(vertices(g) .=> [i for i in 1:length(vertices(g))])

  no_xs = 1000
  xs = [(1.0 / no_xs) * i for i in 1:no_xs]
  xevals = [calculate_x(calculate_xis(x, vertex_map), vertex_map) for x in xs]
  bond_dims = [1, 2, 3, 4, 5, 6]
  f_actual = weierstrass.(a, nterms, xevals)

  p1 = Plots.plot(xevals, f_actual; legend=false, titlefont=font(10, "Computer Modern"))
  xlabel!("x")
  title!(L"W(x) = \sum_{k}\sin(\pi k^{a} x) / (\pi k^{a})")

  mld, vertex_sequence = optimal_vertex_ordering(a, nterms, 1000, 1e-5)
  #vertex_sequence = [i for i in 1:length(vertices(g))]

  ψ = weierstrass_itn(s, vertex_map, a, nterms; cutoff=1e-16)
  p2 = Plots.plot(; titlefont=font(10, "Computer Modern"), yaxis=:log)
  MPS_avg_error = zeros(length(bond_dims))
  MPS_memory_costs = zeros(length(bond_dims))
  for (χindex, χ) in enumerate(bond_dims)
    println("Truncating Down to Bond Dimension $χ")
    ψtrunc = truncate(ψ; maxdim=χ)
    f_itn = zeros(no_xs)
    for (index, x) in enumerate(xs)
      xis = calculate_xis(x, vertex_map)
      eval_point = calculate_x(xis, vertex_map)
      ψproj = get_bitstring_network(ψtrunc, s, xis)
      f_itn[index] = real(ITensors.contract(ψproj)[])
    end
    MPS_avg_error[χindex] = mean(abs.(f_itn - f_actual))
    MPS_memory_costs[χindex] = Base.summarysize(ψtrunc)
    Plots.plot!(xevals, abs.(f_itn - f_actual); label="χ = $χ")
  end
  xlabel!("x")
  ylabel!("|W(x) - f(x, χ)|")
  mld = maxlinkdim(ψ)
  title!("MPS f(x, χ). χmax = $mld"; fontsize=10)

  Random.seed!(999)
  g = NamedGraph(Graphs.SimpleGraph(binary_tree(5)))
  g = rename_vertices(g, Dict(zip(vertices(g), [(v,) for v in vertices(g)])))
  s = siteinds("S=1/2", g)
  vertex_map = Dict(vertices(g) .=> vertex_sequence)
  ψ = weierstrass_itn(s, vertex_map, a, nterms; cutoff=1e-16)
  p3 = Plots.plot(; titlefont=font(10, "Computer Modern"), yaxis=:log)
  Tree_avg_error = zeros(length(bond_dims))
  Tree_memory_costs = zeros(length(bond_dims))

  @show vertex_map
  d = Dict(vertices(g) => [distance_to_leaf(g, v) for v in vertices(g)])
  @show d

  for (χindex, χ) in enumerate(bond_dims)
    println("Truncating Down to Bond Dimension $χ")
    ψtrunc = truncate(ψ; maxdim=χ)
    f_itn = zeros(no_xs)
    for (index, x) in enumerate(xs)
      xis = calculate_xis(x, vertex_map)
      eval_point = calculate_x(xis, vertex_map)
      ψproj = get_bitstring_network(ψtrunc, s, xis)
      f_itn[index] = real(ITensors.contract(ψproj)[])
    end

    Tree_avg_error[χindex] = mean(abs.(f_itn - f_actual))
    Tree_memory_costs[χindex] = Base.summarysize(ψtrunc)

    Plots.plot!(xevals, abs.(f_itn - f_actual); label="χ = $χ")
  end
  xlabel!("x")
  ylabel!("|W(x) - f(x, χ)|")
  mld = maxlinkdim(ψ)
  title!("Binary Tree f(x, χ). χmax = $mld")

  p4 = Plots.plot(
    MPS_memory_costs / (1e6),
    MPS_avg_error;
    labels="MPS",
    yaxis=:log,
    titlefont=font(10, "Computer Modern"),
    xguidefontsize=10,
  )
  Plots.plot!(Tree_memory_costs / (1e6), Tree_avg_error; labels="Tree", yaxis=:log)

  xlabel!("Memory Cost (Mb)")
  title!("Mean(|W(x) - f(x, χ)|)")

  l = @layout [a d; b c]
  plot(p1, p4, p2, p3; layout=l)
  return Plots.savefig("examples/weierstrass_example_output.pdf")
end

main()
