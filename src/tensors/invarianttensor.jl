# invarianttensor.jl
#
# ITensor provides a dense implementation of an AbstractTensor type living
# in an InvariantSpace, i.e. the invariant subspace of the tensor product
# of its index spaces.
# Currenlty only defined for abelian sectors.

#+++++++++++++++++++++++
# InvariantTensor type:
#+++++++++++++++++++++++
# Type definition and constructors:
#-----------------------------------
immutable InvariantTensor{G,S,T,N} <: AbstractTensor{S,InvariantSpace,T,N}
    data::Vector{T}
    space::InvariantSpace{G,S,N}
    _datasectors::Dict{NTuple{N,G},Array{T,N}}
    function InvariantTensor(data::Array{T},space::InvariantSpace{G,S,N})
        if !(length(data)==dim(space) || ndims(data) == dim(space) == 0)
            throw(DimensionMismatch("data not of right size"))
        end
        if promote_type(T,eltype(S))!=eltype(S)
            warn("For a tensor in $(space), the entries should not be of type $(T)")
        end
        _datasectors=Dict{NTuple{N,G},Array{T,N}}()
        ind=1
        for s in sectors(space)
            dims=ntuple(n->dim(space[n],s[n]),N)
            _datasectors[s]=pointer_to_array(pointer(data,ind),dims)
            ind+=prod(dims)
        end
        return new(vec(data),space,_datasectors)
    end
end

# Show method:
#-------------
function Base.show{G,S,T,N}(io::IO,t::InvariantTensor{G,S,T,N})
    print(io," InvariantTensor ∈ $T")
    print(io,"[")
    showcompact(io,t.space.dims)
    println(io,"]:")
    sec = sectors(t)
    if length(sec) == 0
        println(io,t.data[1])
    else
        for s in sectors(t)
            println(io,"$s:")
            Base.showarray(io,t[s];header=false)
            println(io,"")
        end
    end
end

# Basic methods for characterising a tensor:
#--------------------------------------------
space(t::InvariantTensor,ind::Int)=t.space[ind]
space(t::InvariantTensor)=t.space

# General constructors
#---------------------
# with data
tensor{G,S,T,N}(data::Array{T},P::InvariantSpace{G,S,N})=InvariantTensor{G,S,T,N}(data,P)

vdim(dim::Int) = dim == 0 ? () : (dim)

# without data
tensor{T}(::Type{T},P::InvariantSpace)=tensor(Array(T,vdim(dim(P))),P)
tensor(P::InvariantSpace)=tensor(Float64,V)

Base.similar{G,S,T,N}(t::InvariantTensor{G,S},::Type{T},P::InvariantSpace{G,S,N}=space(t))=tensor(similar(t.data,T,vdim(dim(P))),P)
Base.similar{G,S,T,N}(t::InvariantTensor{G,S},::Type{T},P::ProductSpace{S,N})=similar(t,T,invariant(P))
Base.similar{G,S,T}(t::InvariantTensor{G,S},::Type{T},V::S)=similar(t,T,invariant(V))
Base.similar{G,S,N}(t::InvariantTensor{G,S},P::InvariantSpace{G,S,N}=space(t))=similar(t,eltype(t),P)
Base.similar{G,S,N}(t::InvariantTensor{G,S},P::ProductSpace{S,N})=similar(t,eltype(t),P)
Base.similar{G,S}(t::InvariantTensor{G,S},V::S)=similar(t,eltype(t),V)

Base.zero(t::InvariantTensor)=tensor(zero(t.data),space(t))

Base.zeros{T}(::Type{T},P::InvariantSpace)=tensor(zeros(T,vdim(dim(P))),P)
Base.zeros(P::InvariantSpace)=zeros(Float64,P)

Base.rand{T}(::Type{T},P::InvariantSpace)=tensor(rand(T,vdim(dim(P))),P)
Base.rand(P::InvariantSpace)=rand(Float64,P)

function Base.eye{S<:UnitaryRepresentationSpace,T}(::Type{T},::Type{InvariantSpace},V::S)
    t=zeros(T,invariant(V⊗dual(V)))
    for s in sectors(V)
        for n=1:dim(V,s)
            t[(s,conj(s))][n,n] = one(T)
        end
    end
    return t
end
Base.eye{S<:UnitaryRepresentationSpace}(::Type{InvariantSpace},V::S)=eye(Float64,InvariantSpace,V)

Base.eye{G<:Sector,S<:UnitaryRepresentationSpace,T}(::Type{T},P::InvariantSpace{G,S,2})=(P[1]==dual(P[2]) ? eye(T,InvariantSpace,P[1]) : throw(SpaceError("Cannot construct eye-tensor when second space is not the dual of the first space")))
Base.eye{G<:Sector,S<:UnitaryRepresentationSpace}(P::InvariantSpace{G,S,2})=eye(Float64,P)

# TO BE DONE
# # tensors from concatenation
# function tensorcat{S}(catind, X::Tensor{S}...)
#     catind = collect(catind)
#     isempty(catind) && error("catind should not be empty")
#     # length(unique(catdims)) != length(catdims) && error("every dimension should appear only once")

#     nargs = length(X)
#     numindX = map(numind, X)

#     all(n->(n == numindX[1]), numindX) || throw(SpaceError("all tensors should have the same number of indices for concatenation"))

#     numindC = numindX[1]
#     ncatind = setdiff(1:numindC,catind)
#     spaceCvec = Array(S, numindC)
#     for n = 1:numindC
#         spaceCvec[n] = space(X[1],n)
#     end
#     for i = 2:nargs
#         for n in catind
#             spaceCvec[n] = directsum(spaceCvec[n], space(X[i],n))
#         end
#         for n in ncatind
#             spaceCvec[n] == space(X[i],n) || throw(SpaceError("space mismatch for index $n"))
#         end
#     end
#     spaceC = ⊗(spaceCvec...)
#     typeC = mapreduce(eltype, promote_type, X)
#     dataC = zeros(typeC, map(dim,spaceC))

#     offset = zeros(Int,numindC)
#     for i=1:nargs
#         currentdims=ntuple(n->dim(space(X[i],n)),numindC)
#         currentrange=[offset[n]+(1:currentdims[n]) for n=1:numindC]
#         dataC[currentrange...] = X[i].data
#         for n in catind
#             offset[n]+=currentdims[n]
#         end
#     end
#     return tensor(dataC,spaceC)
# end

# Copy and fill tensors:
#------------------------
function Base.copy!(tdest::InvariantTensor,tsource::InvariantTensor)
    # Copies data of tensor tsource to tensor tdest if compatible
    space(tdest)==space(tsource) || throw(SpaceError())
    for s in sectors(tdest)
        copy!(tdest[s],tsource[s])
    end
    return tdest
end
Base.fill!{G,S,T}(tdest::InvariantTensor{G,S,T},value::Number)=fill!(tdest.data,convert(T,value))

# Vectorization:
#----------------
Base.vec(t::InvariantTensor)=t.data
# Convert the non-trivial degrees of freedom in a tensor to a vector to be passed to eigensolvers etc.

@generated function Base.full{G,S,T,N}(t::InvariantTensor{G,S,T,N})
    quote
        dims=@ntuple $N n->dim(space(t,n))
        a=zeros(T,dims)
        for s in sectors(t)
            @nexprs $N n->(r_{n}=to_range(s[n],space(t,n)))
            (@nref $N a r) = t[s]
        end
        return a
    end
end


# Conversion and promotion:
#---------------------------
Base.promote_rule{G,S,T1,T2,N}(::Type{InvariantTensor{G,S,T1,N}},::Type{InvariantTensor{G,S,T2,N}})=InvariantTensor{G,S,promote_type(T1,T2),N}
Base.promote_rule{G,S,T1,T2,N1,N2}(::Type{InvariantTensor{G,S,T1,N1}},::Type{InvariantTensor{G,S,T2,N2}})=InvariantTensor{G,S,promote_type(T1,T2)}
Base.promote_rule{G,S,T1,T2}(::Type{InvariantTensor{G,S,T1}},::Type{InvariantTensor{G,S,T2}})=InvariantTensor{G,S,promote_type(T1,T2)}

Base.promote_rule{G,S,T1,T2,N}(::Type{AbstractTensor{S,InvariantSpace,T1,N}},::Type{InvariantTensor{G,S,T2,N}})=AbstractTensor{S,InvariantSpace,promote_type(T1,T2),N}
Base.promote_rule{G,S,T1,T2,N1,N2}(::Type{AbstractTensor{S,InvariantSpace,T1,N1}},::Type{InvariantTensor{G,S,T2,N2}})=AbstractTensor{S,InveriantSpace,promote_type(T1,T2)}
Base.promote_rule{G,S,T1,T2}(::Type{AbstractTensor{S,InvariantSpace,T1}},::Type{InvariantTensor{G,S,T2}})=AbstractTensor{S,InvariantSpace,promote_type(T1,T2)}

Base.convert{G,S,T,N}(::Type{InvariantTensor{G,S,T,N}},t::InvariantTensor{G,S,T,N})=t
Base.convert{G,S,T1,T2,N}(::Type{InvariantTensor{G,S,T1,N}},t::InvariantTensor{G,S,T2,N})=copy!(similar(t,T1),t)
Base.convert{G,S,T}(::Type{InvariantTensor{G,S,T}},t::InvariantTensor{G,S,T})=t
Base.convert{G,S,T1,T2}(::Type{InvariantTensor{G,S,T1}},t::InvariantTensor{G,S,T2})=copy!(similar(t,T1),t)

Base.float{G,S,T<:AbstractFloat}(t::InvariantTensor{G,S,T})=t
Base.float(t::InvariantTensor)=tensor(float(t.data),space(t))

Base.real{G,S,T<:Real}(t::InvariantTensor{G,S,T})=t
Base.real(t::InvariantTensor)=tensor(real(t.data),space(t))
Base.complex{G,S,T<:Complex}(t::InvariantTensor{G,S,T})=t
Base.complex(t::InvariantTensor)=tensor(complex(t.data),space(t))

for (f,T) in ((:float32,    Float32),
              (:float64,    Float64),
              (:complex64,  Complex64),
              (:complex128, Complex128))
    @eval (Base.$f){G,S}(t::InvariantTensor{G,S,$T}) = t
    @eval (Base.$f)(t::InvariantTensor) = tensor(($f)(t.data),space(t))
end

invariant(t::InvariantTensor)=t

function invariant{G<:Abelian,T,N}(t::Tensor{AbelianSpace{G},T,N})
    for s in sectors(t)
        prod(s)==one(G) || all(t[s].==0) || throw(InexactError())
    end
    tdest=tensor(T,invariant(space(t)))
    for s in sectors(tdest)
        copy!(tdest[s],t[s])
    end
    return tdest
end

# Basic algebra:
#----------------
function Base.conj!(t1::InvariantTensor,t2::InvariantTensor)
    space(t1)==conj(space(t2)) || throw(SpaceError())
    for s in sectors(t2)
        copy!(t1[conj(s)],t2[s])
    end
    conj!(t1.data)
    return t1
end

# transpose inverts order of indices, compatible with graphical notation
function Base.transpose!(tdest::InvariantTensor,tsource::InvariantTensor)
    space(tdest)==space(tsource).' || throw(SpaceError())
    N=numind(tsource)
    for s in sectors(tsource)
        TensorOperations.tensorcopy!(tsource[s],1:N,tdest[reverse(s)],reverse(1:N))
    end
    return tdest
end
function Base.ctranspose!(tdest::InvariantTensor,tsource::InvariantTensor)
    space(tdest)==space(tsource)' || throw(SpaceError())
    N=numind(tsource)
    for s in sectors(tsource)
        TensorOperations.tensorcopy!(tsource[s],1:N,tdest[reverse(map(conj,s))],reverse(1:N))
    end
    conj!(tdest.data)
    return tdest
end

Base.scale!{G,S,T,N}(t1::InvariantTensor{G,S,T,N},t2::InvariantTensor{G,S,T,N},a::Number)=(space(t1)==space(t2) ? scale!(t1.data,t2.data,a) : throw(SpaceError());return t1)
Base.scale!{G,S,T,N}(t1::InvariantTensor{G,S,T,N},a::Number,t2::InvariantTensor{G,S,T,N})=(space(t1)==space(t2) ? scale!(t1.data,a,t2.data) : throw(SpaceError());return t1)

Base.LinAlg.axpy!(a::Number,x::InvariantTensor,y::InvariantTensor)=(space(x)==space(y) ? Base.LinAlg.axpy!(a,x.data,y.data) : throw(SpaceError()); return y)

-(t::InvariantTensor)=tensor(-t.data,space(t))
+(t1::InvariantTensor,t2::InvariantTensor)= space(t1)==space(t2) ? tensor(t1.data+t2.data,space(t1)) : throw(SpaceError())
-(t1::InvariantTensor,t2::InvariantTensor)= space(t1)==space(t2) ? tensor(t1.data-t2.data,space(t1)) : throw(SpaceError())

# Scalar product and norm: only implemented for AbelianSpace
Base.dot{G<:Abelian}(t1::InvariantTensor{G,AbelianSpace{G}},t2::InvariantTensor{G,AbelianSpace{G}})= (space(t1)==space(t2) ? dot(vec(t1),vec(t2)) : throw(SpaceError()))
Base.vecnorm{G<:Abelian}(t::InvariantTensor{G,AbelianSpace{G}})=vecnorm(t.data) # frobenius norm

# Indexing
#----------
# using sectors
Base.getindex{G,S,T,N}(t::InvariantTensor{G,S,T,N},s::NTuple{N,G})=t._datasectors[s]
Base.setindex!{G,S,T,N}(t::InvariantTensor{G,S,T,N},v::Array,s::NTuple{N,G})=(length(v)==length(t[s]) ? copy!(t[s],v) : throw(DimensionMismatch()))
Base.setindex!{G,S,T,N}(t::InvariantTensor{G,S,T,N},v::Number,s::NTuple{N,G})=fill!(t[s],v)

# Tensor Operations
#-------------------
TensorOperations.scalar(t::InvariantTensor)=iscnumber(space(t)) ? t.data[1] : throw(SpaceError("Not a scalar"))

function TensorOperations.add!{CA}(α, A::InvariantTensor, ::Type{Val{CA}}, β, C::InvariantTensor, indCinA)
    # Implements C = β*C + α*permute(op(A))
    NA = numind(A)

    spaceA = CA == :C ? conj(space(A)) : space(A)
    
    for i = 1:NA
        spaceA[indCinA[i]] == space(C,i) || throw(SpaceError("incompatible index spaces of tensors"))
    end

    if NA == 0 #scalars
        C.data[1] = β*C.data[1] + α*A.data[1]
    elseif indCinA == collect(1:NA) #trivial permutation
        if β == 0
            scale!(copy!(C,A), α)
        else
            Base.LinAlg.axpy!(α, A, scale!(C, β))
        end
    else
        for s in sectors(A)
            TensorOperations.add!(α, A[s], Val{CA}, β, C[s[indCinA]], indCinA)
        end
    end

    return C
end

function TensorOperations.trace!{CA}(α, A::InvariantTensor, ::Type{Val{CA}}, β, C::InvariantTensor, indCinA, cindA1, cindA2)
    NA = numind(A)
    NC = numind(C)

    spaceA = CA == :C ? conj(space(A)) : space(A)
    
    for i = 1:NC
        spaceA[indCinA[i]] == space(C,i) || throw(SpaceError("space mismatch"))
    end
    
    for i = 1:div(NA-NC, 2)
        spaceA[cindA1[i]] == dual(spaceA[cindA2[i]]) || throw(SpaceError("space mismatch"))
    end

    if length(indCinA) == 0
        Cscal = pointer_to_array(pointer(C.data, 1), ())
        for sA in sectors(A)
            if sA[cindA1] != conj(sA[cindA2])
                continue
            end
            TensorOperations.trace!(α, A[sA], Val{CA}, β, Cscal, indCinA, cindA1, cindA2)
            β = one(β)
        end
    else
        βdict = [s=>β for s in sectors(C)]
        for sA in sectors(A)
            if sA[cindA1] != conj(sA[cindA2])
                continue
            end
            sC = sA[indCinA]
            TensorOperations.trace!(α, A[sA], Val{CA}, βdict[sC], C[sC], indCinA, cindA1, cindA2)
            βdict[sC] = one(β)
        end
    end
    
    return C
end

function TensorOperations.contract!{CA,CB,ME}(α, A::InvariantTensor, ::Type{Val{CA}}, B::InvariantTensor, ::Type{Val{CB}}, β, C::InvariantTensor, oindA, cindA, oindB, cindB, indCinoAB, ::Type{Val{ME}}=Val{:BLAS})
    # check size compatibility
    spaceA = CA == :C ? conj(space(A)) : space(A)
    spaceB = CB == :C ? conj(space(B)) : space(B)
    spaceC = space(C)

    cspaceA = spaceA[cindA]
    cspaceB = spaceB[cindB]

    ospaceA = spaceA[oindA]
    ospaceB = spaceB[oindB]

    ospaceAB = ospaceA ⊗ ospaceB

    for i = 1:length(cspaceA)
        cspaceA[i] == dual(cspaceB[i]) || throw(SpaceError("A and B have incompatible index spaces"))
    end

    for i in 1:length(indCinoAB)
        spaceC[i] == ospaceAB[indCinoAB[i]] || throw(SpaceError("C has incompatible index space"))
    end

    if length(indCinoAB) == 0
        Cscal = pointer_to_array(pointer(C.data, 1), ())
        for sA in sectors(spaceA), sB in sectors(spaceB)
            if sA[cindA] == conj(sB)[cindB]
                sA_ = (CA==:C?conj(sA):sA)
                sB_ = (CB==:C?conj(sB):sB)
                TensorOperations.contract!(α, A[sA_], Val{CA}, B[sB_], Val{CB}, β, Cscal, oindA, cindA, oindB, cindB, indCinoAB, Val{ME})
                β = one(β)
            end
        end
    else
        βdict = [s=>β for s in sectors(C)]
        for sA in sectors(spaceA), sB in sectors(spaceB)
            if sA[cindA] == conj(sB)[cindB]
                sC = tuple(sA[oindA]..., sB[oindB]...)[indCinoAB]
                sA_ = (CA == :C ? conj(sA) : sA) #undo conjugation for indexing of original tensors
                sB_ = (CB == :C ? conj(sB) : sB)
                TensorOperations.contract!(α, A[sA_], Val{CA}, B[sB_], Val{CB}, βdict[sC], C[sC], oindA, cindA, oindB, cindB, indCinoAB, Val{ME})
                βdict[sC] = one(β)
            end
        end
    end

    return C
end

function TensorOperations.similar_from_indices{T,CA}(::Type{T}, indices, A::InvariantTensor, ::Type{Val{CA}}=Val{:N})
    spaceA = (CA == :C ? conj(space(A)) : space(A))
    return similar(A, T, spaceA[indices])
end

function TensorOperations.similar_from_indices{T,CA,CB}(::Type{T}, indices, A::InvariantTensor, B::InvariantTensor, ::Type{Val{CA}}=Val{:N}, ::Type{Val{CB}}=Val{:N})
    spaceA = (CA == :C ? conj(space(A)) : space(A))
    spaceB = (CB == :C ? conj(space(B)) : space(B))
    spaceAB = spaceA ⊗ spaceB
    return similar(A, T, spaceAB[indices])
end

# Index methods
#---------------
@eval function insertind{G,S}(t::InvariantTensor{G,S},ind::Int,V::S)
    N=numind(t)
    0<=ind<=N || throw(IndexError("index out of range"))
    iscnumber(V) || throw(SpaceError("can only insert index with c-number index space"))
    spacet=space(t)
    newspace=spacet[1:ind] ⊗ V ⊗ spacet[ind+1:N]
    tdest=similar(t,newspace)
    for s in sectors(tdest)
        st=tuple(s[1:ind]...,s[ind+2:N+1]...)
        copy!(tdest[s],t[st])
    end
    return tdest
end
@eval function deleteind(t::InvariantTensor,ind::Int)
    1<=ind<=numind(t) || throw(IndexError("index out of range"))
    iscnumber(space(t,ind)) || throw(SpaceError("can only delete index with c-number index space"))
    spacet=space(t)
    newspace=spacet[1:ind-1] ⊗ spacet[ind+1:N]
    tdest=similar(t,newspace)
    for s in sectors(tsource)
        st=tuple(s[1:ind-1]...,s[ind+1:N]...)
        copy!(tdest[st],t[s])
    end
    return tensor(t.data,newspace)
end


# Methods below are only implemented for AbelianTensor:
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++
typealias AbelianTensor{G<:Abelian,T,N} InvariantTensor{G,AbelianSpace{G},T,N}

# for (S,TT) in ((CartesianSpace,CartesianTensor),(ComplexSpace,ComplexTensor))
#     @eval function fuseind(t::$TT,ind1::Int,ind2::Int,V::$S)
#         N=numind(t)
#         ind2==ind1+1 || throw(IndexError("only neighbouring indices can be fused"))
#         1<=ind1<=N-1 || throw(IndexError("index out of range"))
#         fuse(space(t,ind1),space(t,ind2),V) || throw(SpaceError("index spaces $(space(t,ind1)) and $(space(t,ind2)) cannot be fused to $V"))
#         spacet=space(t)
#         newspace=spacet[1:ind1-1]*V*spacet[ind2+1:N]
#         return tensor(t.data,newspace)
#     end
#     @eval function splitind(t::$TT,ind::Int,V1::$S,V2::$S)
#         1<=ind<=numind(t) || throw(IndexError("index out of range"))
#         fuse(V1,V2,space(t,ind)) || throw(SpaceError("index space $(space(t,ind)) cannot be split into $V1 and $V2"))
#         spacet=space(t)
#         newspace=spacet[1:ind-1]*V1*V2*spacet[ind+1:N]
#         return tensor(t.data,newspace)
#     end
# end

# Factorizations:
#-----------------
function _reorderdata!{G<:Abelian,T,N}(t::AbelianTensor{G,T,N},tmp::Vector{T},n::Int)
    # Reorders data in t, using a temporary array of size dim(t), as preparation for
    # a factorization according to a cut between index n and index n+1. The resulting
    # tensor t is no longer safe to use, as t will no longer be equal to tensor(vec(t),space(t)).

    spacet=space(t)
    leftspace=spacet[1:n]
    rightspace=spacet[n+1:N]
    
    length(tmp)==dim(spacet) || throw(DimensionMismatch())
    
    t2=tensor(tmp,space(t))
    copy!(t2,t)
    
    leftdims=Dict{NTuple{n,G},Int}()
    rightdims=Dict{NTuple{N-n,G},Int}()
    for s in sectors(t2)
        ls=s[1:n]
        rs=s[n+1:N]
        leftdims[ls]=dim(leftspace,ls)
        rightdims[rs]=dim(rightspace,rs)
    end
    
    # Determine sizes of blocks of constant fusing charge
    blockleftdims=Dict{G,Int}()
    blockrightdims=Dict{G,Int}()
    leftoffsets=Dict{NTuple{n,G},Int}()
    rightoffsets=Dict{NTuple{N-n,G},Int}()
    for s in keys(leftdims)
        c=prod(s)
        leftoffsets[s]=get(blockleftdims,c,0)
        blockleftdims[c]=get(blockleftdims,c,0)+leftdims[s]
    end
    for s in keys(rightdims)
        c=conj(prod(s))
        rightoffsets[s]=get(blockrightdims,c,0)
        blockrightdims[c]=get(blockrightdims,c,0)+rightdims[s]
    end
    
    # Build new blocks in t
    blocks=Dict{G,Array{T,2}}()
    offset=0
    for c in keys(blockleftdims)
        blocks[c]=pointer_to_array(pointer(t.data,offset+1),(blockleftdims[c],blockrightdims[c]))
        offset+=length(blocks[c])
    end
    for s in sectors(t2)
        ls=s[1:n]
        rs=s[n+1:N]
        c=prod(ls)
        lo=leftoffsets[ls]
        ro=rightoffsets[rs]
        ld=leftdims[ls]
        rd=rightdims[rs]
        
        src=reshape(t2[s],(ld,rd))
        copy!(blocks[c],lo+(1:ld),ro+(1:rd),src,1:ld,1:rd)
    end
    return blocks, blockleftdims, blockrightdims, leftoffsets, rightoffsets, leftdims, rightdims
end

function svd!{G<:Abelian,T,N}(t::AbelianTensor{G,T,N},n::Int,::NoTruncation)
    # Perform singular value decomposition corresponding to bipartion of the
    # tensor indices into the left indices 1:n and remaining right indices,
    # thereby destroying the original tensor.
    
    spacet=space(t)
    leftspace=spacet[1:n]
    rightspace=spacet[n+1:N]
    truncerr=abs(zero(T))
    ST=typeof(truncerr)
    
    # prepare data
    tmp=similar(vec(t))
    blocks, blockleftdims, blockrightdims, leftoffsets, rightoffsets, leftdims, rightdims = _reorderdata!(t,tmp,n)
    
    # perform singular value decomposition
    dims=Dict{G,Int}()
    facts=Dict{G,Base.LinAlg.SVD{T,ST}}()
    for c in keys(blocks)
        F=svdfact!(blocks[c])
        facts[c]=F
        dims[c]=length(F[:S])
    end
    newspace=AbelianSpace(dims)
    
    # bring output in tensor form
    spaceU=invariant(leftspace*newspace')
    spaceV=invariant(newspace*rightspace)
    spaceS=invariant(newspace*newspace')
    if abs(dim(spaceU)-length(tmp)) < abs(dim(spaceV)-length(tmp))
        dataU=resize!(tmp,dim(spaceU))
        dataV=Array(T,dim(spaceV))
    else
        dataU=Array(T,dim(spaceU))
        dataV=resize!(tmp,dim(spaceV))
    end
    dataS=zeros(ST,dim(spaceS))
    U=tensor(dataU,spaceU)
    V=tensor(dataV,spaceV)
    S=tensor(dataS,spaceS)
    
    for c in sectors(newspace)
        Sblock=S[tuple(c,conj(c))]
        F=facts[c]
        for i=1:dims[c]
            Sblock[i,i]=F[:S][i]
        end
    end
    for s in sectors(t)
        ls=s[1:n]
        rs=s[n+1:N]
        c=prod(ls)
        lo=leftoffsets[ls]
        ro=rightoffsets[rs]
        ld=leftdims[ls]
        rd=rightdims[rs]
        d=dims[c]
        
        if d>0
            Ublock=reshape(U[tuple(ls...,conj(c))],(ld,d))
            Vblock=reshape(V[tuple(c,rs...)],(d,rd))
        
            F=facts[c]
            copy!(Ublock,1:ld,1:d,F[:U],lo+(1:ld),1:d)
            copy!(Vblock,1:d,1:rd,F[:Vt],1:d,ro+(1:rd))
        end
    end
    
    return U,S,V,truncerr
end

function svd!{G<:Abelian,T,N}(t::AbelianTensor{G,T,N},n::Int,trunc::TruncationDimension)
    # Perform singular value decomposition corresponding to bipartion of the
    # tensor indices into the left indices 1:n and remaining right indices,
    # thereby destroying the original tensor.
    spacet=space(t)
    leftspace=spacet[1:n]
    rightspace=spacet[n+1:N]
    truncerr=abs(zero(T))
    ST=typeof(truncerr)
    
    tmp=similar(vec(t))
    blocks, blockleftdims, blockrightdims, leftoffsets, rightoffsets, leftdims, rightdims = _reorderdata!(t,tmp,n)
    
    # perform singular value decomposition
    dims=Dict{G,Int}()
    facts=Dict{G,Base.LinAlg.SVD{T,ST}}()
    for c in keys(blocks)
        F=svdfact!(blocks[c])
        facts[c]=F
        dims[c]=length(F[:S])
    end
    newspace=AbelianSpace(dims)
    
    # perform truncation
    sing=Array(ST,sum(values(dims)))
    ind=1
    for c in keys(facts)
        F=facts[c]
        for sigma in F[:S]
            sing[ind]=sigma
            ind+=1
        end
    end
    sing=sort!(sing,rev=true)
    numsing=length(sing)
    normsing=vecnorm(sing)
    if eps(trunc)!=0
        truncdim=0
        while truncdim<length(sing) && vecnorm(slice(sing,(truncdim+1):numsing))>eps(trunc)*normsing
            truncdim+=1
        end
        truncdim=min(truncdim,dim(trunc))
    else
        truncdim=dim(trunc)
    end
    if truncdim<length(sing)
        truncerr=vecnorm(slice(sing,(truncdim+1):numsing))/normsing
        trunctol=sqrt(sing[truncdim]*sing[truncdim+1])
    else
        trunctol=zero(ST)
    end
    for c in keys(facts)
        F=facts[c]
        dims[c]=sum(F[:S].>trunctol)
    end
    newspace=AbelianSpace(dims)
    
    # bring output in tensor form
    spaceU=invariant(leftspace*newspace')
    spaceV=invariant(newspace*rightspace)
    spaceS=invariant(newspace*newspace')
    if abs(dim(spaceU)-length(tmp)) < abs(dim(spaceV)-length(tmp))
        dataU=resize!(tmp,dim(spaceU))
        dataV=Array(T,dim(spaceV))
    else
        dataU=Array(T,dim(spaceU))
        dataV=resize!(tmp,dim(spaceV))
    end
    dataS=zeros(ST,dim(spaceS))
    U=tensor(dataU,spaceU)
    V=tensor(dataV,spaceV)
    S=tensor(dataS,spaceS)
    
    for c in sectors(newspace)
        Sblock=S[tuple(c,conj(c))]
        F=facts[c]
        for i=1:dims[c]
            Sblock[i,i]=F[:S][i]
        end
    end
    for s in sectors(t)
        ls=s[1:n]
        rs=s[n+1:N]
        c=prod(ls)
        lo=leftoffsets[ls]
        ro=rightoffsets[rs]
        ld=leftdims[ls]
        rd=rightdims[rs]
        d=dims[c]
        
        if d>0
            Ublock=reshape(U[tuple(ls...,conj(c))],(ld,d))
            Vblock=reshape(V[tuple(c,rs...)],(d,rd))
        
            F=facts[c]
            copy!(Ublock,1:ld,1:d,F[:U],lo+(1:ld),1:d)
            copy!(Vblock,1:d,1:rd,F[:Vt],1:d,ro+(1:rd))
        end
    end
    
    return U,S,V,truncerr
end

function svd!{G<:Abelian,T,N}(t::AbelianTensor{G,T,N},n::Int,trunc::TruncationSpace)
    # Perform singular value decomposition corresponding to bipartion of the
    # tensor indices into the left indices 1:n and remaining right indices,
    # thereby destroying the original tensor.
    spacet=space(t)
    leftspace=spacet[1:n]
    rightspace=spacet[n+1:N]
    truncerr=abs(zero(T))
    ST=typeof(truncerr)
    
    tmp=similar(vec(t))
    blocks, blockleftdims, blockrightdims, leftoffsets, rightoffsets, leftdims, rightdims = _reorderdata!(t,tmp,n)
    
    # perform singular value decomposition
    dims=Dict{G,Int}()
    facts=Dict{G,Base.LinAlg.SVD{T,ST}}()
    for c in keys(blocks)
        F=svdfact!(blocks[c])
        facts[c]=F
        dims[c]=length(F[:S])
    end
    newspace=AbelianSpace(dims)
    
    # perform truncation
    sing=Array(ST,sum(values(dims)))
    ind=1
    for c in keys(facts)
        F=facts[c]
        for sigma in F[:S]
            sing[ind]=sigma
            ind+=1
        end
    end
    sing=sort!(sing,rev=true)
    numsing=length(sing)
    normsing=vecnorm(sing)
    if eps(trunc)!=0
        truncdim=0
        while truncdim<length(sing) && vecnorm(slice(sing,(truncdim+1):numsing))>eps(trunc)*normsing
            truncdim+=1
        end
        if truncdim<length(sing)
            trunctol=sqrt(sing[truncdim]*sing[truncdim+1])
        else
            trunctol=zero(ST)
        end
        for c in keys(facts)
            F=facts[c]
            dims[c]=min(sum(F[:S].>=trunctol),dim(space(trunc),c))
        end
    else
        for c in keys(facts)
            F=facts[c]
            dims[c]=min(dims[c],dim(space(trunc),c))
        end
    end
    newspace=AbelianSpace(dims)
    fill!(sing,0)
    ind=1
    for c in keys(facts)
        F=facts[c]
        for i=1:dims[c]
            sing[ind]=F[:S][i]
            ind+=1
        end
    end
    truncerr=vecnorm(sing)/normsing
    
    # bring output in tensor form
    spaceU=invariant(leftspace*newspace')
    spaceV=invariant(newspace*rightspace)
    spaceS=invariant(newspace*newspace')
    if abs(dim(spaceU)-length(tmp)) < abs(dim(spaceV)-length(tmp))
        dataU=resize!(tmp,dim(spaceU))
        dataV=Array(T,dim(spaceV))
    else
        dataU=Array(T,dim(spaceU))
        dataV=resize!(tmp,dim(spaceV))
    end
    dataS=zeros(ST,dim(spaceS))
    U=tensor(dataU,spaceU)
    V=tensor(dataV,spaceV)
    S=tensor(dataS,spaceS)
    
    for c in sectors(newspace)
        Sblock=S[tuple(c,conj(c))]
        F=facts[c]
        for i=1:dims[c]
            Sblock[i,i]=F[:S][i]
        end
    end
    for s in sectors(t)
        ls=s[1:n]
        rs=s[n+1:N]
        c=prod(ls)
        lo=leftoffsets[ls]
        ro=rightoffsets[rs]
        ld=leftdims[ls]
        rd=rightdims[rs]
        d=dims[c]
        
        if d>0
            Ublock=reshape(U[tuple(ls...,conj(c))],(ld,d))
            Vblock=reshape(V[tuple(c,rs...)],(d,rd))
        
            F=facts[c]
            copy!(Ublock,1:ld,1:d,F[:U],lo+(1:ld),1:d)
            copy!(Vblock,1:d,1:rd,F[:Vt],1:d,ro+(1:rd))
        end
    end
    
    return U,S,V,truncerr
end

function svd!{G<:Abelian,T,N}(t::AbelianTensor{G,T,N},n::Int,trunc::TruncationError)
    # Perform singular value decomposition corresponding to bipartion of the
    # tensor indices into the left indices 1:n and remaining right indices,
    # thereby destroying the original tensor.
    spacet=space(t)
    leftspace=spacet[1:n]
    rightspace=spacet[n+1:N]
    truncerr=abs(zero(T))
    ST=typeof(truncerr)
    
    tmp=similar(vec(t))
    blocks, blockleftdims, blockrightdims, leftoffsets, rightoffsets, leftdims, rightdims = _reorderdata!(t,tmp,n)
    
    # perform singular value decomposition
    dims=Dict{G,Int}()
    facts=Dict{G,Base.LinAlg.SVD{T,ST}}()
    for c in keys(blocks)
        F=svdfact!(blocks[c])
        facts[c]=F
        dims[c]=length(F[:S])
    end
    newspace=AbelianSpace(dims)
    
    # perform truncation
    sing=Array(ST,sum(values(dims)))
    ind=1
    for c in keys(facts)
        F=facts[c]
        for sigma in F[:S]
            sing[ind]=sigma
            ind+=1
        end
    end
    sing=sort!(sing,rev=true)
    normsing=vecnorm(sing)
    numsing=length(sing)
    truncdim=0
    while truncdim<length(sing) && vecnorm(slice(sing,(truncdim+1):numsing))>eps(trunc)*normsing
        truncdim+=1
    end
    if truncdim<length(sing)
        truncerr=vecnorm(slice(sing,(truncdim+1):numsing))/normsing
        trunctol=sqrt(sing[truncdim]*sing[truncdim+1])
    else
        trunctol=zero(ST)
    end
    for c in keys(facts)
        F=facts[c]
        dims[c]=sum(F[:S].>=trunctol)
    end
    newspace=AbelianSpace(dims)
    
    # bring output in tensor form
    spaceU=invariant(leftspace*newspace')
    spaceV=invariant(newspace*rightspace)
    spaceS=invariant(newspace*newspace')
    if abs(dim(spaceU)-length(tmp)) < abs(dim(spaceV)-length(tmp))
        dataU=resize!(tmp,dim(spaceU))
        dataV=Array(T,dim(spaceV))
    else
        dataU=Array(T,dim(spaceU))
        dataV=resize!(tmp,dim(spaceV))
    end
    dataS=zeros(ST,dim(spaceS))
    U=tensor(dataU,spaceU)
    V=tensor(dataV,spaceV)
    S=tensor(dataS,spaceS)
    
    for c in sectors(newspace)
        Sblock=S[tuple(c,conj(c))]
        F=facts[c]
        for i=1:dims[c]
            Sblock[i,i]=F[:S][i]
        end
    end
    for s in sectors(t)
        ls=s[1:n]
        rs=s[n+1:N]
        c=prod(ls)
        lo=leftoffsets[ls]
        ro=rightoffsets[rs]
        ld=leftdims[ls]
        rd=rightdims[rs]
        d=dims[c]
        
        if d>0
            Ublock=reshape(U[tuple(ls...,conj(c))],(ld,d))
            Vblock=reshape(V[tuple(c,rs...)],(d,rd))
        
            F=facts[c]
            copy!(Ublock,1:ld,1:d,F[:U],lo+(1:ld),1:d)
            copy!(Vblock,1:d,1:rd,F[:Vt],1:d,ro+(1:rd))
        end
    end
    
    return U,S,V,truncerr
end

function leftorth!{G<:Abelian,T,N}(t::AbelianTensor{G,T,N},n::Int,::NoTruncation)
    # Perform singular value decomposition corresponding to bipartion of the
    # tensor indices into the left indices 1:n and remaining right indices,
    # thereby destroying the original tensor.
    spacet=space(t)
    leftspace=spacet[1:n]
    rightspace=spacet[n+1:N]
    truncerr=abs(zero(T))
    ST=typeof(truncerr)
    
    # prepare data
    tmp=similar(vec(t))
    blocks, blockleftdims, blockrightdims, leftoffsets, rightoffsets, leftdims, rightdims = _reorderdata!(t,tmp,n)
    
    # perform singular value decomposition
    dims=Dict{G,Int}()
    Us=Dict{G,Array{T,2}}()
    Cs=Dict{G,Array{T,2}}()
    for c in keys(blocks)
        if blockleftdims[c]>blockrightdims[c]
            F=qrfact!(blocks[c])
            Us[c]=full(F[:Q])
            Cs[c]=full(F[:R])
        else
            Us[c]=eye(T,blockleftdims[c])
            Cs[c]=blocks[c]
        end
        dims[c]=min(blockleftdims[c],blockrightdims[c])
    end
    newspace=AbelianSpace(dims)
    
    # bring output in tensor form
    spaceU=invariant(leftspace*newspace')
    spaceC=invariant(newspace*rightspace)
    dataU=resize!(tmp,dim(spaceU))
    dataC=Array(T,dim(spaceC))
    U=tensor(dataU,spaceU)
    C=tensor(dataC,spaceC)
    
    for s in sectors(t)
        ls=s[1:n]
        rs=s[n+1:N]
        c=prod(ls)
        lo=leftoffsets[ls]
        ro=rightoffsets[rs]
        ld=leftdims[ls]
        rd=rightdims[rs]
        d=dims[c]
        
        if d>0
            Ublock=reshape(U[tuple(ls...,conj(c))],(ld,d))
            Cblock=reshape(C[tuple(c,rs...)],(d,rd))
        
            copy!(Ublock,1:ld,1:d,Us[c],lo+(1:ld),1:d)
            copy!(Cblock,1:d,1:rd,Cs[c],1:d,ro+(1:rd))
        end
    end
    
    return U,C,truncerr
end

function leftorth!{G<:Abelian,T,N}(t::AbelianTensor{G,T,N},n::Int,trunc::TruncationDimension)
    # Perform singular value decomposition corresponding to bipartion of the
    # tensor indices into the left indices 1:n and remaining right indices,
    # thereby destroying the original tensor.
    spacet=space(t)
    leftspace=spacet[1:n]
    rightspace=spacet[n+1:N]
    truncerr=abs(zero(T))
    ST=typeof(truncerr)
    
    tmp=similar(vec(t))
    blocks, blockleftdims, blockrightdims, leftoffsets, rightoffsets, leftdims, rightdims = _reorderdata!(t,tmp,n)
    
    # perform singular value decomposition
    dims=Dict{G,Int}()
    facts=Dict{G,Base.LinAlg.SVD{T,ST}}()
    Us=Dict{G,Array{T,2}}()
    Cs=Dict{G,Array{T,2}}()
    for c in keys(blocks)
        F=svdfact!(blocks[c])
        facts[c]=F
        dims[c]=length(F[:S])
        Us[c]=F[:U]
        Cs[c]=scale!(F[:S],F[:Vt])
    end
    newspace=AbelianSpace(dims)
    
    # perform truncation
    sing=Array(ST,sum(values(dims)))
    ind=1
    for c in keys(facts)
        F=facts[c]
        for sigma in F[:S]
            sing[ind]=sigma
            ind+=1
        end
    end
    sing=sort!(sing,rev=true)
    numsing=length(sing)
    normsing=vecnorm(sing)
    if eps(trunc)!=0
        truncdim=0
        while truncdim<length(sing) && vecnorm(slice(sing,(truncdim+1):numsing))>eps(trunc)*normsing
            truncdim+=1
        end
        truncdim=min(truncdim,dim(trunc))
    else
        truncdim=dim(trunc)
    end
    if truncdim<length(sing)
        truncerr=vecnorm(slice(sing,(truncdim+1):numsing))/normsing
        trunctol=sqrt(sing[truncdim]*sing[truncdim+1])
    else
        trunctol=zero(ST)
    end
    for c in keys(facts)
        F=facts[c]
        dims[c]=sum(F[:S].>trunctol)
    end
    newspace=AbelianSpace(dims)
    
    # bring output in tensor form
    spaceU=invariant(leftspace*newspace')
    spaceC=invariant(newspace*rightspace)
    dataU=resize!(tmp,dim(spaceU))
    dataC=Array(T,dim(spaceC))
    U=tensor(dataU,spaceU)
    C=tensor(dataC,spaceC)
    
    for s in sectors(t)
        ls=s[1:n]
        rs=s[n+1:N]
        c=prod(ls)
        lo=leftoffsets[ls]
        ro=rightoffsets[rs]
        ld=leftdims[ls]
        rd=rightdims[rs]
        d=dims[c]
        
        if d>0
            Ublock=reshape(U[tuple(ls...,conj(c))],(ld,d))
            Cblock=reshape(C[tuple(c,rs...)],(d,rd))
        
            copy!(Ublock,1:ld,1:d,Us[c],lo+(1:ld),1:d)
            copy!(Cblock,1:d,1:rd,Cs[c],1:d,ro+(1:rd))
        end
    end
    
    return U,C,truncerr
end

function leftorth!{G<:Abelian,T,N}(t::AbelianTensor{G,T,N},n::Int,trunc::TruncationSpace)
    # Perform singular value decomposition corresponding to bipartion of the
    # tensor indices into the left indices 1:n and remaining right indices,
    # thereby destroying the original tensor.
    spacet=space(t)
    leftspace=spacet[1:n]
    rightspace=spacet[n+1:N]
    truncerr=abs(zero(T))
    ST=typeof(truncerr)
    
    tmp=similar(vec(t))
    blocks, blockleftdims, blockrightdims, leftoffsets, rightoffsets, leftdims, rightdims = _reorderdata!(t,tmp,n)
    
    # perform singular value decomposition
    dims=Dict{G,Int}()
    facts=Dict{G,Base.LinAlg.SVD{T,ST}}()
    Us=Dict{G,Array{T,2}}()
    Cs=Dict{G,Array{T,2}}()
    for c in keys(blocks)
        F=svdfact!(blocks[c])
        facts[c]=F
        dims[c]=length(F[:S])
        Us[c]=F[:U]
        Cs[c]=scale!(F[:S],F[:Vt])
    end
    newspace=AbelianSpace(dims)
    
    # perform truncation
    sing=Array(ST,sum(values(dims)))
    ind=1
    for c in keys(facts)
        F=facts[c]
        for sigma in F[:S]
            sing[ind]=sigma
            ind+=1
        end
    end
    sing=sort!(sing,rev=true)
    numsing=length(sing)
    normsing=vecnorm(sing)
    if eps(trunc)!=0
        truncdim=0
        while truncdim<length(sing) && vecnorm(slice(sing,(truncdim+1):numsing))>eps(trunc)*normsing
            truncdim+=1
        end
        if truncdim<length(sing)
            trunctol=sqrt(sing[truncdim]*sing[truncdim+1])
        else
            trunctol=zero(ST)
        end
        for c in keys(facts)
            F=facts[c]
            dims[c]=min(sum(F[:S].>=trunctol),dim(space(trunc),c))
        end
    else
        for c in keys(facts)
            F=facts[c]
            dims[c]=min(dims[c],dim(space(trunc),c))
        end
    end
    newspace=AbelianSpace(dims)
    fill!(sing,0)
    ind=1
    for c in keys(facts)
        F=facts[c]
        for i=1:dims[c]
            sing[ind]=F[:S][i]
            ind+=1
        end
    end
    truncerr=vecnorm(sing)/normsing
    
    # bring output in tensor form
    spaceU=invariant(leftspace*newspace')
    spaceC=invariant(newspace*rightspace)
    dataU=resize!(tmp,dim(spaceU))
    dataC=Array(T,dim(spaceC))
    U=tensor(dataU,spaceU)
    C=tensor(dataC,spaceC)
    
    for s in sectors(t)
        ls=s[1:n]
        rs=s[n+1:N]
        c=prod(ls)
        lo=leftoffsets[ls]
        ro=rightoffsets[rs]
        ld=leftdims[ls]
        rd=rightdims[rs]
        d=dims[c]
        
        if d>0
            Ublock=reshape(U[tuple(ls...,conj(c))],(ld,d))
            Cblock=reshape(C[tuple(c,rs...)],(d,rd))
        
            copy!(Ublock,1:ld,1:d,Us[c],lo+(1:ld),1:d)
            copy!(Cblock,1:d,1:rd,Cs[c],1:d,ro+(1:rd))
        end
    end
    
    return U,C,truncerr
end

function leftorth!{G<:Abelian,T,N}(t::AbelianTensor{G,T,N},n::Int,trunc::TruncationError)
    # Perform singular value decomposition corresponding to bipartion of the
    # tensor indices into the left indices 1:n and remaining right indices,
    # thereby destroying the original tensor.
    spacet=space(t)
    leftspace=spacet[1:n]
    rightspace=spacet[n+1:N]
    truncerr=abs(zero(T))
    ST=typeof(truncerr)
    
    tmp=similar(vec(t))
    blocks, blockleftdims, blockrightdims, leftoffsets, rightoffsets, leftdims, rightdims = _reorderdata!(t,tmp,n)
    
    # perform singular value decomposition
    dims=Dict{G,Int}()
    facts=Dict{G,Base.LinAlg.SVD{T,ST}}()
    Us=Dict{G,Array{T,2}}()
    Cs=Dict{G,Array{T,2}}()
    for c in keys(blocks)
        F=svdfact!(blocks[c])
        facts[c]=F
        dims[c]=length(F[:S])
        Us[c]=F[:U]
        Cs[c]=scale!(F[:S],F[:Vt])
    end
    newspace=AbelianSpace(dims)
    
    # perform truncation
    sing=Array(ST,sum(values(dims)))
    ind=1
    for c in keys(facts)
        F=facts[c]
        for sigma in F[:S]
            sing[ind]=sigma
            ind+=1
        end
    end
    sing=sort!(sing,rev=true)
    numsing=length(sing)
    normsing=vecnorm(sing)
    truncdim=0
    while truncdim<length(sing) && vecnorm(slice(sing,(truncdim+1):numsing))>eps(trunc)*normsing
        truncdim+=1
    end
    if truncdim<length(sing)
        truncerr=vecnorm(slice(sing,(truncdim+1):numsing))/normsing
        trunctol=sqrt(sing[truncdim]*sing[truncdim+1])
    else
        trunctol=zero(ST)
    end
    for c in keys(facts)
        F=facts[c]
        dims[c]=sum(F[:S].>=trunctol)
    end
    newspace=AbelianSpace(dims)
    
    # bring output in tensor form
    spaceU=invariant(leftspace*newspace')
    spaceC=invariant(newspace*rightspace)
    dataU=resize!(tmp,dim(spaceU))
    dataC=Array(T,dim(spaceC))
    U=tensor(dataU,spaceU)
    C=tensor(dataC,spaceC)
    
    for s in sectors(t)
        ls=s[1:n]
        rs=s[n+1:N]
        c=prod(ls)
        lo=leftoffsets[ls]
        ro=rightoffsets[rs]
        ld=leftdims[ls]
        rd=rightdims[rs]
        d=dims[c]
        
        if d>0
            Ublock=reshape(U[tuple(ls...,conj(c))],(ld,d))
            Cblock=reshape(C[tuple(c,rs...)],(d,rd))
        
            copy!(Ublock,1:ld,1:d,Us[c],lo+(1:ld),1:d)
            copy!(Cblock,1:d,1:rd,Cs[c],1:d,ro+(1:rd))
        end
    end
    
    return U,C,truncerr
end

function rightorth!{G<:Abelian,T,N}(t::AbelianTensor{G,T,N},n::Int,::NoTruncation)
    # Perform singular value decomposition corresponding to bipartion of the
    # tensor indices into the left indices 1:n and remaining right indices,
    # thereby destroying the original tensor.
    spacet=space(t)
    leftspace=spacet[1:n]
    rightspace=spacet[n+1:N]
    truncerr=abs(zero(T))
    ST=typeof(truncerr)
    
    # prepare data
    tmp=similar(vec(t))
    t2=tensor(tmp,space(t))
    copy!(t2,t)
    blocks, blockleftdims, blockrightdims, leftoffsets, rightoffsets, leftdims, rightdims = _reorderdata!(t2,t.data,n)
    offset=0
    tblocks=Dict{G,Array{T,2}}()
    for c in keys(blockleftdims)
        tblocks[c]=pointer_to_array(pointer(t.data,offset+1),(blockrightdims[c],blockleftdims[c]))
        Base.transpose!(tblocks[c],blocks[c])
        offset+=length(tblocks[c])
    end
    
    # perform singular value decomposition
    dims=Dict{G,Int}()
    Cs=Dict{G,Array{T,2}}()
    Us=Dict{G,Array{T,2}}()
    for c in keys(blocks)
        if blockleftdims[c]<blockrightdims[c]
            F=qrfact!(tblocks[c])
            Us[c]=full(F[:Q])
            Cs[c]=full(F[:R])
        else
            Us[c]=eye(T,blockrightdims[c])
            Cs[c]=tblocks[c]
        end
        dims[c]=min(blockleftdims[c],blockrightdims[c])
    end
    newspace=AbelianSpace(dims)
    
    # bring output in tensor form
    spaceC=invariant(leftspace*newspace')
    spaceU=invariant(newspace*rightspace)
    dataC=Array(T,dim(spaceC))
    dataU=resize!(tmp,dim(spaceU))
    C=tensor(dataC,spaceC)
    U=tensor(dataU,spaceU)
    
    for s in sectors(t)
        ls=s[1:n]
        rs=s[n+1:N]
        c=prod(ls)
        lo=leftoffsets[ls]
        ro=rightoffsets[rs]
        ld=leftdims[ls]
        rd=rightdims[rs]
        d=dims[c]
        
        if d>0
            Cblock=reshape(C[tuple(ls...,conj(c))],(ld,d))
            Ublock=reshape(U[tuple(c,rs...)],(d,rd))
        
            Base.transpose!(Cblock,slice(Cs[c],1:d,lo+(1:ld)))
            Base.transpose!(Ublock,slice(Us[c],ro+(1:rd),1:d))
        end
    end
    
    return C,U,truncerr
end

function rightorth!{G<:Abelian,T,N}(t::AbelianTensor{G,T,N},n::Int,trunc::TruncationDimension)
    # Perform singular value decomposition corresponding to bipartion of the
    # tensor indices into the left indices 1:n and remaining right indices,
    # thereby destroying the original tensor.
    spacet=space(t)
    leftspace=spacet[1:n]
    rightspace=spacet[n+1:N]
    truncerr=abs(zero(T))
    ST=typeof(truncerr)
    
    tmp=similar(vec(t))
    blocks, blockleftdims, blockrightdims, leftoffsets, rightoffsets, leftdims, rightdims = _reorderdata!(t,tmp,n)
    
    # perform singular value decomposition
    dims=Dict{G,Int}()
    facts=Dict{G,Base.LinAlg.SVD{T,ST}}()
    Cs=Dict{G,Array{T,2}}()
    Us=Dict{G,Array{T,2}}()
    for c in keys(blocks)
        F=svdfact!(blocks[c])
        facts[c]=F
        dims[c]=length(F[:S])
        Cs[c]=scale!(F[:U],F[:S])
        Us[c]=F[:Vt]
    end
    newspace=AbelianSpace(dims)
    
    # perform truncation
    sing=Array(ST,sum(values(dims)))
    ind=1
    for c in keys(facts)
        F=facts[c]
        for sigma in F[:S]
            sing[ind]=sigma
            ind+=1
        end
    end
    sing=sort!(sing,rev=true)
    numsing=length(sing)
    normsing=vecnorm(sing)
    if eps(trunc)!=0
        truncdim=0
        while truncdim<length(sing) && vecnorm(slice(sing,(truncdim+1):numsing))>eps(trunc)*normsing
            truncdim+=1
        end
        truncdim=min(truncdim,dim(trunc))
    else
        truncdim=dim(trunc)
    end
    if truncdim<length(sing)
        truncerr=vecnorm(slice(sing,(truncdim+1):numsing))/normsing
        trunctol=sqrt(sing[truncdim]*sing[truncdim+1])
    else
        trunctol=zero(ST)
    end
    for c in keys(facts)
        F=facts[c]
        dims[c]=sum(F[:S].>trunctol)
    end
    newspace=AbelianSpace(dims)
    
    # bring output in tensor form
    spaceC=invariant(leftspace*newspace')
    spaceU=invariant(newspace*rightspace)
    dataC=Array(T,dim(spaceC))
    dataU=resize!(tmp,dim(spaceU))
    C=tensor(dataC,spaceC)
    U=tensor(dataU,spaceU)
    
    for s in sectors(t)
        ls=s[1:n]
        rs=s[n+1:N]
        c=prod(ls)
        lo=leftoffsets[ls]
        ro=rightoffsets[rs]
        ld=leftdims[ls]
        rd=rightdims[rs]
        d=dims[c]
        
        if d>0
            Cblock=reshape(C[tuple(ls...,conj(c))],(ld,d))
            Ublock=reshape(U[tuple(c,rs...)],(d,rd))
        
            copy!(Cblock,1:ld,1:d,Cs[c],lo+(1:ld),1:d)
            copy!(Ublock,1:d,1:rd,Us[c],1:d,ro+(1:rd))
        end
    end
    
    return C,U,truncerr
end

function rightorth!{G<:Abelian,T,N}(t::AbelianTensor{G,T,N},n::Int,trunc::TruncationSpace)
    # Perform singular value decomposition corresponding to bipartion of the
    # tensor indices into the left indices 1:n and remaining right indices,
    # thereby destroying the original tensor.
    spacet=space(t)
    leftspace=spacet[1:n]
    rightspace=spacet[n+1:N]
    truncerr=abs(zero(T))
    ST=typeof(truncerr)
    
    tmp=similar(vec(t))
    blocks, blockleftdims, blockrightdims, leftoffsets, rightoffsets, leftdims, rightdims = _reorderdata!(t,tmp,n)
    
    # perform singular value decomposition
    dims=Dict{G,Int}()
    facts=Dict{G,Base.LinAlg.SVD{T,ST}}()
    Cs=Dict{G,Array{T,2}}()
    Us=Dict{G,Array{T,2}}()
    for c in keys(blocks)
        F=svdfact!(blocks[c])
        facts[c]=F
        dims[c]=length(F[:S])
        Cs[c]=scale!(F[:U],F[:S])
        Us[c]=F[:Vt]
    end
    newspace=AbelianSpace(dims)
    
    # perform truncation
    sing=Array(ST,sum(values(dims)))
    ind=1
    for c in keys(facts)
        F=facts[c]
        for sigma in F[:S]
            sing[ind]=sigma
            ind+=1
        end
    end
    sing=sort!(sing,rev=true)
    numsing=length(sing)
    normsing=vecnorm(sing)
    if eps(trunc)!=0
        truncdim=0
        while truncdim<length(sing) && vecnorm(slice(sing,(truncdim+1):numsing))>eps(trunc)*normsing
            truncdim+=1
        end
        if truncdim<length(sing)
            trunctol=sqrt(sing[truncdim]*sing[truncdim+1])
        else
            trunctol=zero(ST)
        end
        for c in keys(facts)
            F=facts[c]
            dims[c]=min(sum(F[:S].>=trunctol),dim(space(trunc),c))
        end
    else
        for c in keys(facts)
            F=facts[c]
            dims[c]=min(dims[c],dim(space(trunc),c))
        end
    end
    newspace=AbelianSpace(dims)
    fill!(sing,0)
    ind=1
    for c in keys(facts)
        F=facts[c]
        for i=1:dims[c]
            sing[ind]=F[:S][i]
            ind+=1
        end
    end
    truncerr=vecnorm(sing)/normsing
    
    # bring output in tensor form
    spaceC=invariant(leftspace*newspace')
    spaceU=invariant(newspace*rightspace)
    dataC=Array(T,dim(spaceC))
    dataU=resize!(tmp,dim(spaceU))
    C=tensor(dataC,spaceC)
    U=tensor(dataU,spaceU)
    
    for s in sectors(t)
        ls=s[1:n]
        rs=s[n+1:N]
        c=prod(ls)
        lo=leftoffsets[ls]
        ro=rightoffsets[rs]
        ld=leftdims[ls]
        rd=rightdims[rs]
        d=dims[c]
        
        if d>0
            Cblock=reshape(C[tuple(ls...,conj(c))],(ld,d))
            Ublock=reshape(U[tuple(c,rs...)],(d,rd))
        
            copy!(Cblock,1:ld,1:d,Cs[c],lo+(1:ld),1:d)
            copy!(Ublock,1:d,1:rd,Us[c],1:d,ro+(1:rd))
        end
    end
    
    return C,U,truncerr
end

function rightorth!{G<:Abelian,T,N}(t::AbelianTensor{G,T,N},n::Int,trunc::TruncationError)
    # Perform singular value decomposition corresponding to bipartion of the
    # tensor indices into the left indices 1:n and remaining right indices,
    # thereby destroying the original tensor.
    spacet=space(t)
    leftspace=spacet[1:n]
    rightspace=spacet[n+1:N]
    truncerr=abs(zero(T))
    ST=typeof(truncerr)
    
    tmp=similar(vec(t))
    blocks, blockleftdims, blockrightdims, leftoffsets, rightoffsets, leftdims, rightdims = _reorderdata!(t,tmp,n)
    
    # perform singular value decomposition
    dims=Dict{G,Int}()
    facts=Dict{G,Base.LinAlg.SVD{T,ST}}()
    Cs=Dict{G,Array{T,2}}()
    Us=Dict{G,Array{T,2}}()
    for c in keys(blocks)
        F=svdfact!(blocks[c])
        facts[c]=F
        dims[c]=length(F[:S])
        Cs[c]=scale!(F[:U],F[:S])
        Us[c]=F[:Vt]
    end
    newspace=AbelianSpace(dims)
    
    # perform truncation
    sing=Array(ST,sum(values(dims)))
    ind=1
    for c in keys(facts)
        F=facts[c]
        for sigma in F[:S]
            sing[ind]=sigma
            ind+=1
        end
    end
    sing=sort!(sing,rev=true)
    numsing=length(sing)
    normsing=vecnorm(sing)
    truncdim=0
    while truncdim<length(sing) && vecnorm(slice(sing,(truncdim+1):numsing))>eps(trunc)*normsing
        truncdim+=1
    end
    if truncdim<length(sing)
        truncerr=vecnorm(slice(sing,(truncdim+1):numsing))/normsing
        trunctol=sqrt(sing[truncdim]*sing[truncdim+1])
    else
        trunctol=zero(ST)
    end
    for c in keys(facts)
        F=facts[c]
        dims[c]=sum(F[:S].>=trunctol)
    end
    newspace=AbelianSpace(dims)
    
    # bring output in tensor form
    spaceC=invariant(leftspace*newspace')
    spaceU=invariant(newspace*rightspace)
    dataC=Array(T,dim(spaceC))
    dataU=resize!(tmp,dim(spaceU))
    C=tensor(dataC,spaceC)
    U=tensor(dataU,spaceU)
    
    for s in sectors(t)
        ls=s[1:n]
        rs=s[n+1:N]
        c=prod(ls)
        lo=leftoffsets[ls]
        ro=rightoffsets[rs]
        ld=leftdims[ls]
        rd=rightdims[rs]
        d=dims[c]
        
        if d>0
            Cblock=reshape(C[tuple(ls...,conj(c))],(ld,d))
            Ublock=reshape(U[tuple(c,rs...)],(d,rd))
        
            copy!(Cblock,1:ld,1:d,Cs[c],lo+(1:ld),1:d)
            copy!(Ublock,1:d,1:rd,Us[c],1:d,ro+(1:rd))
        end
    end
    
    return C,U,truncerr
end

# # Methods below are only implemented for CartesianMatrix or ComplexMatrix:
# #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# typealias ComplexMatrix{T} ComplexTensor{T,2}
# typealias CartesianMatrix{T} CartesianTensor{T,2}
#
# function Base.pinv(t::Union{ComplexMatrix,CartesianMatrix})
#     # Compute pseudo-inverse
#     spacet=space(t)
#     data=copy(t.data)
#     leftdim=dim(spacet[1])
#     rightdim=dim(spacet[2])
#
#     F=svdfact!(data)
#     Sinv=F[:S]
#     for k=1:length(Sinv)
#         if Sinv[k]>eps(Sinv[1])*max(leftdim,rightdim)
#             Sinv[k]=one(Sinv[k])/Sinv[k]
#         end
#     end
#     data=F[:V]*scale(F[:S],F[:U]')
#     return tensor(data,spacet')
# end
#
# function Base.eig(t::Union{ComplexMatrix,CartesianMatrix})
#     # Compute eigenvalue decomposition.
#     spacet=space(t)
#     spacet[1] == spacet[2]' || throw(SpaceError("eigenvalue factorization only exists if left and right index space are dual"))
#     data=copy(t.data)
#
#     F=eigfact!(data)
#
#     Lambda=tensor(diagm(F[:values]),spacet)
#     V=tensor(F[:vectors],spacet)
#     return Lambda, V
# end
#
# function Base.inv(t::Union{ComplexMatrix,CartesianMatrix})
#     # Compute inverse.
#     spacet=space(t)
#     spacet[1] == spacet[2]' || throw(SpaceError("inverse only exists if left and right index space are dual"))
#
#     return tensor(inv(t.data),spacet)
# end
