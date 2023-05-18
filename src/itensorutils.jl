using ITensors

"""Add two ITensors with different index sets. Currently we require that they have the same number of indices and the same
number of unique indices"""
function add_itensors(A::ITensor, B::ITensor)
  @assert length(inds(A)) == length(inds(B))

  cinds = commoninds(A, B)
  Auinds = uniqueinds(A, B)
  Buinds = uniqueinds(B, A)
  @assert length(Auinds) == length(Buinds)

  Ap, Bp = permute(A, vcat(cinds, Auinds)), permute(B, vcat(cinds, Buinds))

  Ainds = inds(Ap)
  Binds = inds(Bp)

  A_array = Array(Ap, Ainds)
  B_array = Array(Bp, Binds)

  A_type = eltype(A_array)
  B_type = eltype(B_array)
  @assert A_type == B_type
  extended_dims = [dim(Auinds[i]) + dim(Buinds[i]) for i in 1:length(Auinds)]
  Apb_newdims = Tuple(vcat([dim(i) for i in cinds], extended_dims))
  out_array = zeros(A_type, Apb_newdims)
  A_ind = Tuple(vcat([..], [1:dim(i) for i in Auinds]))
  out_array[A_ind...] = A_array

  B_ind = Tuple(
    vcat(
      [..],
      [(dim(Auinds[i]) + 1):(dim(Auinds[i]) + dim(Buinds[i])) for i in 1:length(Auinds)],
    ),
  )
  out_array[B_ind...] = B_array

  out_inds = vcat(
    cinds, [Index(extended_dims[i], tags(Auinds[i])) for i in 1:length(Auinds)]
  )

  return ITensor(out_array, out_inds)
end
