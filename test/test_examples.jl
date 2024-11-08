@eval module $(gensym())
using ITensorNumericalAnalysis: ITensorNumericalAnalysis
using Test: @testset

@testset "Test examples" begin
  example_files = [
    "2d_laplace_solver.jl",
    "construct_multi_dimensional_function.jl",
    "fredholm_solver.jl"
  ]
  @testset "Test $example_file" for example_file in example_files
    include(joinpath(pkgdir(ITensorNumericalAnalysis), "examples", example_file))
  end
end
end
