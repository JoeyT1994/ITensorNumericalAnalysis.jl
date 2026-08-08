using ITensors: ITensors, Arrow, Index, dim, dir, id, plev, space, tags

function ITensors.Index(
    pivs::Vector{Vector{P}}; 
    dir::Arrow = ITensors.Out, 
    tags = "", 
    plev = 0
) where {P<:Pair}
    return Index(
        rand(ITensors.index_id_rng(), ITensors.IDType), 
        pivs, 
        dir, 
        tags, 
        plev
    )
end

const PivotIndex = Index{Vector{Vector{P}}} where {P<:Pair}

ITensors.dim(i::PivotIndex) = length(space(i))

ITensors.hasqns(i::PivotIndex) = false

haspivots(i::PivotIndex) = true
haspivots(i::Index) = false

function Base.show(io::IO, i::PivotIndex)
    idstr = "$(id(i) % 1000)"
    if length(tags(i)) > 0
        print(
            io,
            "(PI|dim=$(dim(i))|id=$(idstr)|\"$(ITensors.TagSets.tagstring(ITensors.tags(i)))\")$(ITensors.primestring(ITensors.plev(i)))",
        )
    else
        print(io, "(PI|dim=$(dim(i))|id=$(idstr))$(ITensors.primestring(ITensors.plev(i)))")
    end
end