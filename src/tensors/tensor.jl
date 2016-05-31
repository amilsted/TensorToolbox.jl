# tensor.jl
#
# Tensor provides a dense implementation of an AbstractTensor type without any
# symmetry assumptions, i.e. it describes tensors living in the full tensor
# product space of its index spaces.

#++++++++++++++
# Tensor type:
#++++++++++++++
# Type definition and constructors:
#-----------------------------------
immutable Tensor{S,T,N} <: AbstractTensor{S,ProductSpace,T,N}
    data::Array{T,N}
    space::ProductSpace{S,N}
    function Tensor(data::Array{T},space::ProductSpace{S,N})
        if length(data)!=dim(space)
            throw(DimensionMismatch("data not of right size"))
        end
        if promote_type(T,eltype(S))!=eltype(S)
            warn("For a tensor in $(space), the entries should not be of type $(T)")
        end
        return new(reshape(data,map(dim,space)),space)
    end
end

# Show method:
#-------------
function Base.show{S,T,N}(io::IO,t::Tensor{S,T,N})
    print(io," Tensor ∈ $T")
    print(io,"[")
    for n=1:N
        n==1 || print(io, " ⊗ ")
        show(io,space(t,n))
    end
    println(io,"]:")
    Base.showarray(io,t.data;header=false)
end

# Basic methods for characterising a tensor:
#--------------------------------------------
space(t::Tensor,ind::Integer)=t.space[ind]
space(t::Tensor)=t.space

# General constructors
#---------------------
# with data
tensor{T<:Real,N}(data::Array{T,N})=Tensor{CartesianSpace,T,N}(data,mapreduce(CartesianSpace,⊗,size(data)))
function tensor{T<:Complex}(data::Array{T,1})
    warning("for complex array, consider specifying Euclidean index spaces")
    Tensor{ComplexEuclideanSpace,T,1}(data,⊗(ComplexSpace(size(data,1))))
end
function tensor{T<:Complex}(data::Array{T,2})
    warning("for complex array, consider specifying Euclidean index spaces")
    Tensor{ComplexEuclideanSpace,T,2}(data,ComplexSpace(size(data,1))⊗ComplexSpace(size(data,2))')
end

tensor{S,T,N}(data::Array{T},P::ProductSpace{S,N})=Tensor{S,T,N}(data,P)
tensor(data::Array,V::ElementarySpace)=tensor(data,⊗(V))

# without data
tensor{T}(::Type{T},P::ProductSpace)=tensor(Array(T,dim(P)),P)
tensor{T}(::Type{T},V::IndexSpace)=tensor(T,⊗(V))
tensor(V::Union{ProductSpace,IndexSpace})=tensor(Float64,V)

Base.similar{S,T,N}(t::Tensor{S},::Type{T},P::ProductSpace{S,N}=space(t))=tensor(similar(t.data,T,dim(P)),P)
Base.similar{S,T}(t::Tensor{S},::Type{T},V::S)=similar(t,T,⊗(V))
Base.similar{S,N}(t::Tensor{S},P::ProductSpace{S,N}=space(t))=similar(t,eltype(t),P)
Base.similar{S}(t::Tensor{S},V::S)=similar(t,eltype(t),V)

Base.zero(t::Tensor)=tensor(zero(t.data),space(t))

Base.zeros{T}(::Type{T},P::ProductSpace)=tensor(zeros(T,dim(P)),P)
Base.zeros{T}(::Type{T},V::IndexSpace)=zeros(T,⊗(V))
Base.zeros(V::Union{ProductSpace,IndexSpace})=zeros(Float64,V)

Base.rand{T}(::Type{T},P::ProductSpace)=tensor(rand(T,dim(P)),P)
Base.rand{T}(::Type{T},V::IndexSpace)=rand(T,⊗(V))
Base.rand(V::Union{ProductSpace,IndexSpace})=rand(Float64,V)

Base.eye{S<:ElementarySpace,T}(::Type{T},::Type{ProductSpace},V::S)=tensor(eye(T,dim(V)),V⊗dual(V))
Base.eye{S<:ElementarySpace}(::Type{ProductSpace},V::S)=eye(Float64,ProductSpace,V)

Base.eye{S<:ElementarySpace,T}(::Type{T},P::ProductSpace{S,2})=(P[1]==dual(P[2]) ? eye(T,ProductSpace,P[1]) : throw(SpaceError("Cannot construct eye-tensor when second space is not the dual of the first space")))
Base.eye{S<:ElementarySpace}(P::ProductSpace{S,2})=eye(Float64,P)

# tensors from concatenation
function tensorcat{S}(catind, X::Tensor{S}...)
    catind = collect(catind)
    isempty(catind) && error("catind should not be empty")
    # length(unique(catdims)) != length(catdims) && error("every dimension should appear only once")

    nargs = length(X)
    numindX = map(numind, X)

    all(n->(n == numindX[1]), numindX) || throw(SpaceError("all tensors should have the same number of indices for concatenation"))

    numindC = numindX[1]
    ncatind = setdiff(1:numindC,catind)
    spaceCvec = Array(S, numindC)
    for n = 1:numindC
        spaceCvec[n] = space(X[1],n)
    end
    for i = 2:nargs
        for n in catind
            spaceCvec[n] = directsum(spaceCvec[n], space(X[i],n))
        end
        for n in ncatind
            spaceCvec[n] == space(X[i],n) || throw(SpaceError("space mismatch for index $n"))
        end
    end
    spaceC = ⊗(spaceCvec...)
    typeC = mapreduce(eltype, promote_type, X)
    dataC = zeros(typeC, map(dim,spaceC))

    offset = zeros(Int,numindC)
    for i=1:nargs
        currentdims=ntuple(n->dim(space(X[i],n)),numindC)
        currentrange=[offset[n]+(1:currentdims[n]) for n=1:numindC]
        dataC[currentrange...] = X[i].data
        for n in catind
            offset[n]+=currentdims[n]
        end
    end
    return tensor(dataC,spaceC)
end

# Copy and fill tensors:
#------------------------
function Base.copy!(tdest::Tensor,tsource::Tensor)
    # Copies data of tensor tsource to tensor tdest if compatible
    space(tdest)==space(tsource) || throw(SpaceError("tensor spaces don't match"))
    copy!(tdest.data,tsource.data)
    return tdest
end
Base.fill!{S,T}(tdest::Tensor{S,T},value::Number)=fill!(tdest.data,convert(T,value))

# Vectorization:
#----------------
Base.vec(t::Tensor)=vec(t.data)
# Convert the non-trivial degrees of freedom in a tensor to a vector to be passed to eigensolvers etc.

# Conversion and promotion:
#---------------------------
Base.full(t::Tensor)=t.data

Base.promote_rule{S,T1,T2,N}(::Type{Tensor{S,T1,N}},::Type{Tensor{S,T2,N}})=Tensor{S,promote_type(T1,T2),N}
Base.promote_rule{S,T1,T2,N1,N2}(::Type{Tensor{S,T1,N1}},::Type{Tensor{S,T2,N2}})=Tensor{S,promote_type(T1,T2)}
Base.promote_rule{S,T1,T2}(::Type{Tensor{S,T1}},::Type{Tensor{S,T2}})=Tensor{S,promote_type(T1,T2)}

Base.promote_rule{S,T1,T2,N}(::Type{AbstractTensor{S,ProductSpace,T1,N}},::Type{Tensor{S,T2,N}})=AbstractTensor{S,ProductSpace,promote_type(T1,T2),N}
Base.promote_rule{S,T1,T2,N1,N2}(::Type{AbstractTensor{S,ProductSpace,T1,N1}},::Type{Tensor{S,T2,N2}})=AbstractTensor{S,ProductSpace,promote_type(T1,T2)}
Base.promote_rule{S,T1,T2}(::Type{AbstractTensor{S,ProductSpace,T1}},::Type{Tensor{S,T2}})=AbstractTensor{S,ProductSpace,promote_type(T1,T2)}

Base.convert{S,T,N}(::Type{Tensor{S,T,N}},t::Tensor{S,T,N})=t
Base.convert{S,T1,T2,N}(::Type{Tensor{S,T1,N}},t::Tensor{S,T2,N})=copy!(similar(t,T1),t)
Base.convert{S,T}(::Type{Tensor{S,T}},t::Tensor{S,T})=t
Base.convert{S,T1,T2}(::Type{Tensor{S,T1}},t::Tensor{S,T2})=copy!(similar(t,T1),t)

Base.float{S,T<:AbstractFloat}(t::Tensor{S,T})=t
Base.float(t::Tensor)=tensor(float(t.data),space(t))

Base.real{S,T<:Real}(t::Tensor{S,T})=t
Base.real(t::Tensor)=tensor(real(t.data),space(t))
Base.complex{S,T<:Complex}(t::Tensor{S,T})=t
Base.complex(t::Tensor)=tensor(complex(t.data),space(t))

for (f,T) in ((:float32,    Float32),
              (:float64,    Float64),
              (:complex64,  Complex64),
              (:complex128, Complex128))
    @eval (Base.$f){S}(t::Tensor{S,$T}) = t
    @eval (Base.$f)(t::Tensor) = tensor(($f)(t.data),space(t))
end

# Basic algebra:
#----------------
function Base.conj!(t1::Tensor,t2::Tensor)
    space(t1)==conj(space(t2)) || throw(SpaceError())
    copy!(t1.data,t2.data)
    conj!(t1.data)
    return t1
end

# transpose inverts order of indices, compatible with graphical notation
function Base.transpose!(tdest::Tensor,tsource::Tensor)
    space(tdest)==space(tsource).' || throw(SpaceError())
    N=numind(tsource)
    TensorOperations.tensorcopy!(tsource.data,1:N,tdest.data,reverse(1:N))
    return tdest
end
function Base.ctranspose!(tdest::Tensor,tsource::Tensor)
    space(tdest)==space(tsource)' || throw(SpaceError())
    N=numind(tsource)
    TensorOperations.tensorcopy!(tsource.data,1:N,tdest.data,reverse(1:N))
    conj!(tdest.data)
    return tdest
end

Base.scale!{S,T,N}(t1::Tensor{S,T,N},t2::Tensor{S,T,N},a::Number)=(space(t1)==space(t2) ? scale!(t1.data,t2.data,a) : throw(SpaceError());return t1)
Base.scale!{S,T,N}(t1::Tensor{S,T,N},a::Number,t2::Tensor{S,T,N})=(space(t1)==space(t2) ? scale!(t1.data,a,t2.data) : throw(SpaceError());return t1)

Base.LinAlg.axpy!(a::Number,x::Tensor,y::Tensor)=(space(x)==space(y) ? Base.LinAlg.axpy!(a,x.data,y.data) : throw(SpaceError()); return y)

-(t::Tensor)=tensor(-t.data,space(t))
+(t1::Tensor,t2::Tensor)= space(t1)==space(t2) ? tensor(t1.data+t2.data,space(t1)) : throw(SpaceError())
-(t1::Tensor,t2::Tensor)= space(t1)==space(t2) ? tensor(t1.data-t2.data,space(t1)) : throw(SpaceError())

# Scalar product and norm: only valid for EuclideanSpace
Base.dot{S<:EuclideanSpace}(t1::Tensor{S},t2::Tensor{S})= (space(t1)==space(t2) ? dot(vec(t1),vec(t2)) : throw(SpaceError()))
Base.vecnorm{S<:EuclideanSpace}(t::Tensor{S})=vecnorm(t.data) # frobenius norm

# Indexing
#----------
# # linear indexing using ProductBasisVector
# Base.getindex{S,T,N}(t::Tensor{S,T,N},b::ProductBasisVector{N,S})=getindex(t.data,Base.to_index(b))
# Base.setindex!{S,T,N}(t::Tensor{S,T,N},value,b::ProductBasisVector{N,S})=setindex!(t.data,value,Base.to_index(b))

@generated function Base.getindex{G,T,N}(t::Tensor{AbelianSpace{G},T,N},s::NTuple{N,G})
    quote
        @nexprs $N n->(r_n=to_range(s[n],space(t,n)))
        return @ncall $N slice t.data r
    end
end

Base.setindex!{G,T,N}(t::Tensor{AbelianSpace{G},T,N},v::Array,s::NTuple{N,G})=(length(v)==length(t[s]) ? copy!(t[s],v) : throw(DimensionMismatch()))
Base.setindex!{G,T,N}(t::Tensor{AbelianSpace{G},T,N},v::Number,s::NTuple{N,G})=fill!(t[s],v)

# Tensor Operations
#-------------------
TensorOperations.scalar(t::Tensor)=iscnumber(space(t)) ? t.data[1] : throw(SpaceError("Not a scalar"))

function TensorOperations.add!{CA}(α, A::Tensor, ::Type{Val{CA}}, β, C::Tensor, indCinA)
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
        TensorOperations.add!(α, A.data, Val{CA}, β, C.data, indCinA)
    end

    return C
end

function TensorOperations.trace!{CA}(α, A::Tensor, ::Type{Val{CA}}, β, C::Tensor, indCinA, cindA1, cindA2)
    NA = numind(A)
    NC = numind(C)

    spaceA = CA == :C ? conj(space(A)) : space(A)
    
    for i = 1:NC
        spaceA[indCinA[i]] == space(C,i) || throw(SpaceError("space mismatch"))
    end
    
    for i = 1:div(NA-NC, 2)
        spaceA[cindA1[i]] == dual(spaceA[cindA2[i]]) || throw(SpaceError("space mismatch"))
    end

    TensorOperations.trace!(α, A.data, Val{CA}, β, C.data, indCinA, cindA1, cindA2)
    
    return C
end

function TensorOperations.contract!{CA,CB,ME}(α, A::Tensor, ::Type{Val{CA}}, B::Tensor, ::Type{Val{CB}}, β, C::Tensor, oindA, cindA, oindB, cindB, indCinoAB, ::Type{Val{ME}}=Val{:BLAS})
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
        TensorOperations.contract!(α, A.data, Val{CA}, B.data, Val{CB}, β, Cscal, oindA, cindA, oindB, cindB, indCinoAB, Val{ME})
    else
        TensorOperations.contract!(α, A.data, Val{CA}, B.data, Val{CB}, β, C.data, oindA, cindA, oindB, cindB, indCinoAB, Val{ME})
    end

    return C
end

function TensorOperations.similar_from_indices{T,CA}(::Type{T}, indices, A::Tensor, ::Type{Val{CA}}=Val{:N})
    spaceA = CA == :C ? conj(space(A)) : space(A)
    return similar(A, T, spaceA[indices])
end

function TensorOperations.similar_from_indices{T,CA,CB}(::Type{T}, indices, A::Tensor, B::Tensor, 
                                                        ::Type{Val{CA}}=Val{:N}, ::Type{Val{CB}}=Val{:N})
    spaceA = CA == :C ? conj(space(A)) : space(A)
    spaceB = CB == :C ? conj(space(B)) : space(B)
    spaceAB = spaceA ⊗ spaceB
    return similar(A, T, spaceAB[indices])
end

# Methods below are only implemented for CartesianTensor or ComplexTensor:
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
typealias ComplexTensor{T,N} Tensor{ComplexSpace,T,N}
typealias CartesianTensor{T,N} Tensor{CartesianSpace,T,N}

# Index methods
#---------------
@eval function insertind{S}(t::Tensor{S},ind::Int,V::S)
    N=numind(t)
    0<=ind<=N || throw(IndexError("index out of range"))
    iscnumber(V) || throw(SpaceError("can only insert index with c-number index space"))
    spacet=space(t)
    newspace=spacet[1:ind] ⊗ V ⊗ spacet[ind+1:N]
    return tensor(t.data,newspace)
end
@eval function deleteind(t::Tensor,ind::Int)
    N=numind(t)
    1<=ind<=N || throw(IndexError("index out of range"))
    iscnumber(space(t,ind)) || throw(SpaceError("can only delete index with c-number index space"))
    spacet=space(t)
    newspace=spacet[1:ind-1] ⊗ spacet[ind+1:N]
    return tensor(t.data,newspace)
end

for (S,TT) in ((CartesianSpace,CartesianTensor),(ComplexSpace,ComplexTensor))
    @eval function fuseind(t::$TT,ind1::Int,ind2::Int,V::$S)
        N=numind(t)
        ind2==ind1+1 || throw(IndexError("only neighbouring indices can be fused"))
        1<=ind1<=N-1 || throw(IndexError("index out of range"))
        fuse(space(t,ind1),space(t,ind2),V) || throw(SpaceError("index spaces $(space(t,ind1)) and $(space(t,ind2)) cannot be fused to $V"))
        spacet=space(t)
        newspace=spacet[1:ind1-1]*V*spacet[ind2+1:N]
        return tensor(t.data,newspace)
    end
    @eval function splitind(t::$TT,ind::Int,V1::$S,V2::$S)
        1<=ind<=numind(t) || throw(IndexError("index out of range"))
        fuse(V1,V2,space(t,ind)) || throw(SpaceError("index space $(space(t,ind)) cannot be split into $V1 and $V2"))
        spacet=space(t)
        newspace=spacet[1:ind-1]*V1*V2*spacet[ind+1:N]
        return tensor(t.data,newspace)
    end
end

# Factorizations:
#-----------------
for (S,TT) in ((CartesianSpace,CartesianTensor),(ComplexSpace,ComplexTensor))
    @eval function svd!(t::$TT,n::Int,::NoTruncation)
        # Perform singular value decomposition corresponding to bipartion of the
        # tensor indices into the left indices 1:n and remaining right indices,
        # thereby destroying the original tensor.
        N=numind(t)
        spacet=space(t)
        leftspace=spacet[1:n]
        rightspace=spacet[n+1:N]
        leftdim=dim(leftspace)
        rightdim=dim(rightspace)
        data=reshape(t.data,(leftdim,rightdim))
        F=svdfact!(data)
        newdim=length(F[:S])
        newspace=$S(newdim)
        U=tensor(F[:U],leftspace*newspace')
        S=tensor(diagm(F[:S]),newspace*newspace')
        V=tensor(F[:Vt],newspace*rightspace)
        return U,S,V,abs(zero(eltype(t)))
    end

    @eval function svd!(t::$TT,n::Int,trunc::Union{TruncationDimension,TruncationSpace})
        # Truncate rank corresponding to bipartition into left indices 1:n
        # and remain right indices, based on singular value decomposition,
        # thereby destroying the original tensor.
        # Truncation parameters are given as keyword arguments: trunctol should
        # always be one of the possible arguments for specifying truncation, but
        # truncdim can be replaced with different parameters for other types of tensors.

        N=numind(t)
        spacet=space(t)
        leftspace=spacet[1:n]
        rightspace=spacet[n+1:N]
        leftdim=dim(leftspace)
        rightdim=dim(rightspace)
        dim(trunc) >= min(leftdim,rightdim) && return svd!(t,n)
        data=reshape(t.data,(leftdim,rightdim))
        F=svdfact!(data)
        sing=F[:S]

        # compute truncation dimension
        if eps(trunc)!=0
            sing=F[:S]
            normsing=vecnorm(sing)
            truncdim=0
            while truncdim<length(sing) && vecnorm(sing[(truncdim+1):end])>eps(trunc)*normsing
                truncdim+=1
            end
            truncdim=min(truncdim,dim(trunc))
        else
            truncdim=dim(trunc)
        end
        newspace=$S(truncdim)

        # truncate
        truncerr=vecnorm(sing[(truncdim+1):end])/vecnorm(sing)
        U=tensor(F[:U][:,1:truncdim],leftspace*newspace')
        S=tensor(diagm(sing[1:truncdim]),newspace*newspace')
        V=tensor(F[:Vt][1:truncdim,:],newspace*rightspace)
        return U,S,V,truncerr
    end

    @eval function svd!(t::$TT,n::Int,trunc::TruncationError)
        # Truncate rank corresponding to bipartition into left indices 1:n
        # and remain right indices, based on singular value decomposition,
        # thereby destroying the original tensor.
        # Truncation parameters are given as keyword arguments: trunctol should
        # always be one of the possible arguments for specifying truncation, but
        # truncdim can be replaced with different parameters for other types of tensors.

        N=numind(t)
        spacet=space(t)
        leftspace=spacet[1:n]
        rightspace=spacet[n+1:N]
        leftdim=dim(leftspace)
        rightdim=dim(rightspace)
        data=reshape(t.data,(leftdim,rightdim))
        F=svdfact!(data)

        # find truncdim based on trunctolinfo
        sing=F[:S]
        normsing=vecnorm(sing)
        truncdim=0
        while truncdim<length(sing) && vecnorm(sing[(truncdim+1):end])>eps(trunc)*normsing
            truncdim+=1
        end
        newspace=$S(truncdim)

        # truncate
        truncerr=vecnorm(sing[(truncdim+1):end])/normsing
        U=tensor(F[:U][:,1:truncdim],leftspace*newspace')
        S=tensor(diagm(sing[1:truncdim]),newspace*newspace')
        V=tensor(F[:Vt][1:truncdim,:],newspace*rightspace)
        return U,S,V,truncerr
    end

    @eval function leftorth!(t::$TT,n::Int,::NoTruncation)
        # Create orthogonal basis U for indices 1:n, and remainder C for right
        # indices, thereby destroying the original tensor.
        # Decomposition should be unique, such that it always returns the same
        # result for the same input tensor t. UC = QR is fastest but only unique
        # after correcting for phases.
        N=numind(t)
        spacet=space(t)
        leftspace=spacet[1:n]
        rightspace=spacet[n+1:N]
        leftdim=dim(leftspace)
        rightdim=dim(rightspace)
        data=reshape(t.data,(leftdim,rightdim))
        if leftdim>rightdim
            newdim=rightdim
            tau=zeros(eltype(data),(newdim,))
            Base.LinAlg.LAPACK.geqrf!(data,tau)

            C=zeros(eltype(t),(newdim,newdim))
            for j in 1:newdim
                for i in 1:j
                    @inbounds C[i,j]=data[i,j]
                end
            end
            Base.LinAlg.LAPACK.orgqr!(data,tau)
            U=data
            for i=1:newdim
                tau[i]=sign(C[i,i])
            end
            scale!(U,tau)
            scale!(conj!(tau),C)
        else
            newdim=leftdim
            C=data
            U=eye(eltype(data),newdim)
        end

        newspace=$S(newdim)
        return tensor(U,leftspace*newspace'), tensor(C,newspace*rightspace), abs(zero(eltype(t)))
    end

    @eval function leftorth!(t::$TT,n::Int,trunc::Union{TruncationDimension,TruncationSpace})
        # Truncate rank corresponding to bipartition into left indices 1:n
        # and remain right indices, based on singular value decomposition,
        # thereby destroying the original tensor.
        # Truncation parameters are given as keyword arguments: trunctol should
        # always be one of the possible arguments for specifying truncation, but
        # truncdim can be replaced with different parameters for other types of tensors.

        N=numind(t)
        spacet=space(t)
        leftspace=spacet[1:n]
        rightspace=spacet[n+1:N]
        leftdim=dim(leftspace)
        rightdim=dim(rightspace)
        dim(trunc) >= min(leftdim,rightdim) && return leftorth!(t,n)
        data=reshape(t.data,(leftdim,rightdim))
        F=svdfact!(data)
        sing=F[:S]

        # compute truncation dimension
        if eps(trunc)!=0
            sing=F[:S]
            normsing=vecnorm(sing)
            truncdim=0
            while truncdim<length(sing) && vecnorm(sing[(truncdim+1):end])>eps(trunc)*normsing
                truncdim+=1
            end
            truncdim=min(truncdim,dim(trunc))
        else
            truncdim=dim(trunc)
        end
        newspace=$S(truncdim)

        # truncate
        truncerr=vecnorm(sing[(truncdim+1):end])/vecnorm(sing)
        U=tensor(F[:U][:,1:truncdim],leftspace*newspace')
        C=tensor(scale!(sing[1:truncdim],F[:Vt][1:truncdim,:]),newspace*rightspace)
        return U,C,truncerr
    end

    @eval function leftorth!(t::$TT,n::Int,trunc::TruncationError)
        # Truncate rank corresponding to bipartition into left indices 1:n
        # and remain right indices, based on singular value decomposition,
        # thereby destroying the original tensor.
        # Truncation parameters are given as keyword arguments: trunctol should
        # always be one of the possible arguments for specifying truncation, but
        # truncdim can be replaced with different parameters for other types of tensors.

        N=numind(t)
        spacet=space(t)
        leftspace=spacet[1:n]
        rightspace=spacet[n+1:N]
        leftdim=dim(leftspace)
        rightdim=dim(rightspace)
        data=reshape(t.data,(leftdim,rightdim))
        F=svdfact!(data)

        # compute truncation dimension
        sing=F[:S]
        normsing=vecnorm(sing)
        truncdim=0
        while truncdim<length(sing) && vecnorm(sing[(truncdim+1):end])>eps(trunc)*normsing
            truncdim+=1
        end
        newspace=$S(truncdim)

        # truncate
        truncerr=vecnorm(sing[(truncdim+1):end])/vecnorm(sing)
        U=tensor(F[:U][:,1:truncdim],leftspace*newspace')
        C=tensor(scale!(sing[1:truncdim],F[:Vt][1:truncdim,:]),newspace*rightspace)

        return U,C,truncerr
    end

    @eval function rightorth!(t::$TT,n::Int,::NoTruncation)
        # Create orthogonal basis U for right indices, and remainder C for left
        # indices. Decomposition should be unique, such that it always returns the
        # same result for the same input tensor t. CU = LQ is typically fastest but only
        # unique after correcting for phases.
        N=numind(t)
        spacet=space(t)
        leftspace=spacet[1:n]
        rightspace=spacet[n+1:N]
        leftdim=dim(leftspace)
        rightdim=dim(rightspace)
        data=reshape(t.data,(leftdim,rightdim))
        if leftdim<rightdim
            newdim=leftdim
            tau=zeros(eltype(data),(newdim,))
            datat=transpose(data)
            Base.LinAlg.LAPACK.geqrf!(datat,tau)

            C=zeros(eltype(t),(newdim,newdim))
            for j in 1:newdim
                for i in 1:j
                    @inbounds C[j,i]=datat[i,j]
                end
            end
            Base.LinAlg.LAPACK.orgqr!(datat,tau)
            Base.transpose!(data,datat)
            U=data
            
            for i=1:newdim
                tau[i]=sign(C[i,i])
            end
            scale!(tau,U)
            scale!(C,conj!(tau))
        else
            newdim=rightdim
            C=data
            U=eye(eltype(data),newdim)
        end

        newspace=$S(newdim)
        return tensor(C,leftspace ⊗ dual(newspace)), tensor(U,newspace ⊗ rightspace), abs(zero(eltype(t)))
    end

    @eval function rightorth!(t::$TT,n::Int,trunc::Union{TruncationDimension,TruncationSpace})
        # Truncate rank corresponding to bipartition into left indices 1:n
        # and remain right indices, based on singular value decomposition,
        # thereby destroying the original tensor.
        # Truncation parameters are given as keyword arguments: trunctol should
        # always be one of the possible arguments for specifying truncation, but
        # truncdim can be replaced with different parameters for other types of tensors.

        N=numind(t)
        spacet=space(t)
        leftspace=spacet[1:n]
        rightspace=spacet[n+1:N]
        leftdim=dim(leftspace)
        rightdim=dim(rightspace)
        dim(trunc) >= min(leftdim,rightdim) && return rightorth!(t,n)
        data=reshape(t.data,(leftdim,rightdim))
        F=svdfact!(data)
        sing=F[:S]

        # compute truncation dimension
        if eps(trunc)!=0
            sing=F[:S]
            normsing=vecnorm(sing)
            truncdim=0
            while truncdim<length(sing) && vecnorm(sing[(truncdim+1):end])>eps(trunc)*normsing
                truncdim+=1
            end
            truncdim=min(truncdim,dim(trunc))
        else
            truncdim=dim(trunc)
        end
        newspace=$S(truncdim)

        # truncate
        truncerr=vecnorm(sing[(truncdim+1):end])/vecnorm(sing)
        C=tensor(scale!(F[:U][:,1:truncdim],sing[1:truncdim]),leftspace*newspace')
        U=tensor(F[:Vt][1:truncdim,:],newspace*rightspace)
        return C,U,truncerr
    end

    @eval function rightorth!(t::$TT,n::Int,trunc::TruncationError)
        # Truncate rank corresponding to bipartition into left indices 1:n
        # and remain right indices, based on singular value decomposition,
        # thereby destroying the original tensor.
        # Truncation parameters are given as keyword arguments: trunctol should
        # always be one of the possible arguments for specifying truncation, but
        # truncdim can be replaced with different parameters for other types of tensors.

        N=numind(t)
        spacet=space(t)
        leftspace=spacet[1:n]
        rightspace=spacet[n+1:N]
        leftdim=dim(leftspace)
        rightdim=dim(rightspace)
        data=reshape(t.data,(leftdim,rightdim))
        F=svdfact!(data)

        # compute truncation dimension
        sing=F[:S]
        normsing=vecnorm(sing)
        truncdim=0
        while truncdim<length(sing) && vecnorm(sing[(truncdim+1):end])>eps(trunc)*normsing
            truncdim+=1
        end
        newspace=$S(truncdim)

        # truncate
        truncerr=vecnorm(sing[(truncdim+1):end])/vecnorm(sing)
        C=tensor(scale!(F[:U][:,1:truncdim],sing[1:truncdim]),leftspace*newspace')
        U=tensor(F[:Vt][1:truncdim,:],newspace*rightspace)
        return C,U,truncerr
    end
end

# Methods below are only implemented for CartesianMatrix or ComplexMatrix:
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
typealias ComplexMatrix{T} ComplexTensor{T,2}
typealias CartesianMatrix{T} CartesianTensor{T,2}

function Base.pinv(t::Union{ComplexMatrix,CartesianMatrix})
    # Compute pseudo-inverse
    spacet=space(t)
    data=copy(t.data)
    leftdim=dim(spacet[1])
    rightdim=dim(spacet[2])

    F=svdfact!(data)
    Sinv=F[:S]
    for k=1:length(Sinv)
        if Sinv[k]>eps(Sinv[1])*max(leftdim,rightdim)
            Sinv[k]=one(Sinv[k])/Sinv[k]
        end
    end
    data=F[:V]*scale(F[:S],F[:U]')
    return tensor(data,spacet')
end

function Base.eig(t::Union{ComplexMatrix,CartesianMatrix})
    # Compute eigenvalue decomposition.
    spacet=space(t)
    spacet[1] == spacet[2]' || throw(SpaceError("eigenvalue factorization only exists if left and right index space are dual"))
    data=copy(t.data)

    F=eigfact!(data)

    Lambda=tensor(diagm(F[:values]),spacet)
    V=tensor(F[:vectors],spacet)
    return Lambda, V
end

function Base.inv(t::Union{ComplexMatrix,CartesianMatrix})
    # Compute inverse.
    spacet=space(t)
    spacet[1] == spacet[2]' || throw(SpaceError("inverse only exists if left and right index space are dual"))

    return tensor(inv(t.data),spacet)
end
