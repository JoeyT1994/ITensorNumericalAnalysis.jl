using ITensors: ITensor
using TensorNetworkQuantumSimulator

" Naive integration of a function in all dimensions ∫₀¹f({r})d{r} "
function integrate(
        tnf::TensorNetworkFunction; alg = default_contraction_alg(tnf), take_sum = false, kwargs...
    )
    tnf = copy(tnf)
    s = indexmap(tnf)
    c = take_sum ? 1.0 : (1.0 / base(s))
    for v in vertices(tnf)
        indices = siteinds(s, v)
        setindex_preserve!(tnf, tnf[v] * ITensor(eltype(tnf[v]), c, indices...), v)
    end
    return contract(tensornetwork(tnf); alg, kwargs...)
end

" Naive integration of a operator applied to a function in all dimensions ∫₀¹ (operator*f)({r})d{r} "
function integrate(
        operator::AbstractTensorNetwork,
        tnf::TensorNetworkFunction;
        alg = default_contraction_alg(tnf),
        take_sum = false,
        kwargs...,
    )
    s = indexmap(tnf)
    b = base(s)
    sinds = siteinds(s)
    g = graph(tnf)
    c = take_sum ? 1.0 : (1.0 / base(s))
    op = copy(operator)
    for v in vertices(g)
        ∑v = ITensors.ITensor([c for i in 1:b], prime(only(sinds[v])))
        setindex_preserve!(op, noprime(op[v] * ∑v), v)
    end
    op = TensorNetworkState(op, sinds)
    return inner(op, TensorNetworkState(tensornetwork(tnf), sinds); alg, kwargs...)
end

""" Partial integration of function over specified dimensions. By default reduce the resulting network down to a new, smaller one """
function partial_integrate(
        tnf::TensorNetworkFunction, dims::Vector{Int}; merge_vertices = true, take_sum = false
    )
    s = indexmap(tnf)
    new_imap = copy(s)
    tnf = copy(tensornetwork(tnf))
    c = take_sum ? 1.0 : (1.0 / base(s))
    for v in dimension_vertices(s, dims)
        sinds_dim = filter(i -> dimension(s, i) ∈ dims, siteinds(s, v))
        for sind in sinds_dim
            setindex_preserve!(tnf, tnf[v] * ITensor(eltype(tnf[v]), c, sind), v)
            new_imap = rem_index(new_imap, sind)
        end
    end
    if merge_vertices
        tnf = merge_internal_tensors(tnf)
    end
    return TensorNetworkFunction(tnf, new_imap)
end
