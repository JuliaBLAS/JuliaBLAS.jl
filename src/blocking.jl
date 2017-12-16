using SIMD

#struct BlockInfo{T}
#    MC::LinAlg.BlasInt
#    KC:LinAlg.BlasInt
#    A::BLASPtr{T}
#    B::BLASPtr{T}
#end

const MC = 384
const KC = 384
const NC = 4096

const MR = 8
const NR = 6

const _A  = Vector{Float64}(MC*KC)
const _B  = Vector{Float64}(KC*NC)
const _C  = Vector{Float64}(MR*NR)

const _AB = Vector{Float64}(MR*NR)



# Packing algorithm is based on ulmBLAS
# https://apfel.mathematik.uni-ulm.de/~lehn/sghpc/day08/page02.html
function pack_MRxk(k::Integer, A::Array{Float64}, Aoffset::Integer, incRowA::Integer,
                   incColA::Integer, buffer::Array{Float64}, boffset::Integer)
    @inbounds begin
        for j = 1:k
            @simd for i = 1:MR
                buffer[boffset + i] = A[Aoffset + (i - 1)*incRowA + 1]
            end
            boffset += MR
            Aoffset += incColA
        end
    end
end

function pack_A(mc::Integer, kc::Integer, A::Array{Float64}, Aoffset::Integer,
                incRowA::Integer, incColA::Integer, buffer::Array{Float64})
    @inbounds begin
        mp, _mr = divrem(mc, MR)

        boffset = 0
        for i = 1:mp
            pack_MRxk(kc, A, Aoffset, incRowA, incColA, buffer, boffset)
            boffset += kc*MR
            Aoffset += MR*incRowA
        end
        if _mr > 0
            for j = 1:kc
                for i = 1:_mr
                    buffer[boffset + i] = A[Aoffset + (i - 1)*incRowA + 1]
                end
                for i = _mr:MR
                    buffer[boffset + i] = 0.0
                end
                boffset += MR
                Aoffset += incColA
            end
        end
    end
end

function pack_kxNR(k::Integer, B::Array{Float64}, Boffset::Integer, incRowB::Integer,
                   incColB::Integer, buffer::Array{Float64}, boffset::Integer)
    @inbounds begin
        for i = 1:k
            for j = 1:NR
                buffer[boffset + j] = B[Boffset + (j - 1)*incColB + 1]
            end
            boffset += NR
            Boffset += incRowB
        end
    end
end

function pack_B(kc::Integer, nc::Integer, B::Array{Float64}, Boffset::Integer,
                incRowB::Integer, incColB::Integer, buffer::Array{Float64})
    @inbounds begin
        np, _nr = divrem(nc, NR)

        boffset = 0
        for j = 1:np
            pack_kxNR(kc, B, Boffset, incRowB, incColB, buffer, boffset)
            boffset += kc*NR
            Boffset += NR*incColB
        end
        if _nr > 0
            for i = 1:kc
                for j = 1:_nr
                    buffer[boffset + j] = B[Boffset + (j - 1)*incColB + 1]
                end
                for j = _nr + 1:NR
                    buffer[boffset + j] = 0.0
                end
                boffset += NR
                Boffset += incRowB
            end
        end
    end
end

const cl = "~{esi},~{rax},~{rbx},~{rcx}"*string([",~{xmm$i}" for i = 0:15]...)
const asms =
  "\"movl      \$0,      %esi    \n\t"*
    "movq      \$1,      %rax    \n\t"*
    "movq      \$2,      %rbx    \n\t"*
    "movq      \$3,      %rcx    \n\t"*
    "movapd    (%rax),   %xmm0   \n\t"*
    "movapd  16(%rax),   %xmm1   \n\t"*
    "movapd    (%rbx),   %xmm2   \n\t"*
    "xorpd     %xmm8,    %xmm8   \n\t"*
    "xorpd     %xmm9,    %xmm9   \n\t"*
    "xorpd     %xmm10,   %xmm10  \n\t"*
    "xorpd     %xmm11,   %xmm11  \n\t"*
    "xorpd     %xmm12,   %xmm12  \n\t"*
    "xorpd     %xmm13,   %xmm13  \n\t"*
    "xorpd     %xmm14,   %xmm14  \n\t"*
    "xorpd     %xmm15,   %xmm15  \n\t"*
    "xorpd     %xmm3,    %xmm3   \n\t"*
    "xorpd     %xmm4,    %xmm4   \n\t"*
    "xorpd     %xmm5,    %xmm5   \n\t"*
    "xorpd     %xmm6,    %xmm6   \n\t"*
    "xorpd     %xmm7,    %xmm7   \n\t"*
    "testl     %esi,     %esi    \n\t"*
    # Adding ${:uid} at the end of label can fix the error
    # ```
    # error: invalid symbol redefinition
    # LLVM ERROR: Error parsing inline asm
    # ```
    # as referenced in
    # http://llvm.org/docs/LangRef.html#inline-assembler-expressions
    "je        .DWRITEBACK\${:uid}       \n\t"*
    ".DLOOP\${:uid}:                     \n\t"*
    "addpd     %xmm3,  %xmm12    \n\t"*
    "movapd  16(%rbx), %xmm3     \n\t"*
    "addpd     %xmm6,  %xmm13    \n\t"*
    "movapd    %xmm2,  %xmm6     \n\t"*
    "pshufd \$\$78,%xmm2,%xmm4   \n\t"*
    "mulpd     %xmm0,  %xmm2     \n\t"*
    "mulpd     %xmm1,  %xmm6     \n\t"*
    "addpd     %xmm5,  %xmm14    \n\t"*
    "addpd     %xmm7,  %xmm15    \n\t"*
    "movapd    %xmm4,  %xmm7     \n\t"*
    "mulpd     %xmm0,  %xmm4     \n\t"*
    "mulpd     %xmm1,  %xmm7     \n\t"*
    "addpd     %xmm2,  %xmm8     \n\t"*
    "movapd  32(%rbx), %xmm2     \n\t"*
    "addpd     %xmm6,  %xmm9     \n\t"*
    "movapd    %xmm3,  %xmm6     \n\t"*
    "pshufd \$\$78,%xmm3, %xmm5  \n\t"*
    "mulpd     %xmm0,  %xmm3     \n\t"*
    "mulpd     %xmm1,  %xmm6     \n\t"*
    "addpd     %xmm4,  %xmm10    \n\t"*
    "addpd     %xmm7,  %xmm11    \n\t"*
    "movapd    %xmm5,  %xmm7     \n\t"*
    "mulpd     %xmm0,  %xmm5     \n\t"*
    "movapd  32(%rax), %xmm0     \n\t"*
    "mulpd     %xmm1,  %xmm7     \n\t"*
    "movapd  48(%rax), %xmm1     \n\t"*
    "addq      \$\$32,   %rax    \n\t"*
    "addq      \$\$32,   %rbx    \n\t"*
    "decl      %esi              \n\t"*
    "jne      .DLOOP\${:uid}             \n\t"*
    "addpd    %xmm3,   %xmm12    \n\t"*
    "addpd    %xmm6,   %xmm13    \n\t"*
    "addpd    %xmm5,   %xmm14    \n\t"*
    "addpd    %xmm7,   %xmm15    \n\t"*
    ".DWRITEBACK\${:uid}:                \n\t"*
    "movlpd   %xmm8,    (%rcx)   \n\t"*
    "movhpd   %xmm10,  8(%rcx)   \n\t"*
    "movlpd   %xmm9,  16(%rcx)   \n\t"*
    "movhpd   %xmm11, 24(%rcx)   \n\t"*
    "addq     \$\$32,    %rcx    \n\t"*
    "movlpd   %xmm10,   (%rcx)   \n\t"*
    "movhpd   %xmm8,   8(%rcx)   \n\t"*
    "movlpd   %xmm11, 16(%rcx)   \n\t"*
    "movhpd   %xmm9,  24(%rcx)   \n\t"*
    "addq     \$\$32,    %rcx    \n\t"*
    "movlpd   %xmm12,   (%rcx)   \n\t"*
    "movhpd   %xmm14,  8(%rcx)   \n\t"*
    "movlpd   %xmm13, 16(%rcx)   \n\t"*
    "movhpd   %xmm15, 24(%rcx)   \n\t"*
    "addq     \$\$32,    %rcx    \n\t"*
    "movlpd   %xmm14,   (%rcx)   \n\t"*
    "movhpd   %xmm12,  8(%rcx)   \n\t"*
    "movlpd   %xmm15, 16(%rcx)   \n\t"*
    "movhpd   %xmm13, 24(%rcx)   \n\t\", \"r,r,r,r,$cl\"(i32 %0, i64 %1, i64 %2, i64 %3)"

@inline function gemm_micro_kernel(kc::Integer, α::Float64, A::Array{Float64}, Aoffset::Integer,
    B::Array{Float64}, Boffset::Integer, β::Float64, C::Array{Float64}, Coffset::Integer,
    incRowC::Integer, incColC::Integer)
    @inbounds begin
        #Base.llvmcall("""
        #    call void asm $asms
        #    ret void""",
        #Void,
        #Tuple{Cint, UInt, UInt, UInt},
        #Cint(kc), UInt(pointer(A, Aoffset + 1)), UInt(pointer(B, Boffset + 1)), UInt(pointer(_AB)))
        _ker!(Val{MR}, Val{NR}, kc, pointer(A, Aoffset + 1), pointer(B, Boffset + 1), pointer(_AB))

        #  Update C <- beta*C
        if β == 0.0
            for j = 1:NR
                for i =1:MR
                    C[Coffset + (i - 1)*incRowC + (j - 1)*incColC + 1] = 0.0
                end
            end
        elseif β != 1.0
            for j = 1:NR
                for i = 1:MR
                    C[Coffset + (i - 1)*incRowC + (j - 1)*incColC + 1] *= β
                end
            end
        end

        #  Update C <- C + alpha*AB (note: the case alpha==0.0 was already treated in
        #                                  the above layer dgemm_nn)
        if α == 1.0
            for j = 1:NR
                for i = 1:MR
                    C[Coffset + (i - 1)*incRowC + (j - 1)*incColC + 1] += _AB[i + (j - 1)*MR]
                end
            end
        else
            for j = 1:NR
                for i = 1:MR
                    C[Coffset + (i - 1)*incRowC + (j - 1)*incColC + 1] += α*_AB[i + (j - 1)*MR]
                end
            end
        end
    end
end

function geaxpy(m::Integer, n::Integer, α::Float64, X::Array{Float64}, incRowX::Integer,
    incColX::Integer, Y::Array{Float64}, Yoffset::Integer, incRowY::Integer, incColY::Integer)

    if α != 1.0
        for j = 1:n
            for i = 1:m
                Y[Yoffset + (i - 1)*incRowY + (j - 1)*incColY + 1] = Y[(i - 1)*incRowY + (j - 1)*incColY + 1] + α*X[(i - 1)*incRowX + (j - 1)*incColX + 1]
            end
        end
    else
        for j = 1:n
            for i = 1:m
                Y[Yoffset + (i - 1)*incRowY + (j - 1)*incColY + 1] = Y[(i - 1)*incRowY + (j - 1)*incColY + 1] + X[(i - 1)*incRowX + (j - 1)*incColX + 1]
            end
        end
    end
end

function gescal(m::Integer,
                n::Integer,
                α::Float64,
                X::Array{Float64},
                Xoffset::Integer,
                incRowX::Integer,
                incColX::Integer)

    if α != 0.0
        for j = 1:n
            for i = 1:m
                X[Xoffset + (i - 1)*incRowX + (j - 1)*incColX + 1] *= α
            end
        end
    else
        for j = 1:n
            for i = 1:m
                X[Xoffset + (i - 1)*incRowX + (j - 1)*incColX + 1] = 0.0
            end
        end
    end
end

function gemm_macro_kernel(mc::Integer, nc::Integer, kc::Integer, α::Float64,
    β::Float64, C::Array{Float64}, Coffset::Integer, incRowC::Integer, incColC::Integer)

    mp = div(mc + MR - 1, MR)
    np = div(nc + NR - 1, NR)

    _mr = mc % MR
    _nr = nc % NR

    for j = 1:np
        nr = (j != np || _nr == 0) ? NR : _nr

        for i = 1:mp
            mr = (i != mp || _mr == 0) ? MR : _mr

            if mr == MR && nr == NR
                gemm_micro_kernel(kc,
                    α,
                    _A,
                    (i - 1)*kc*MR,
                    _B,
                    (j - 1)*kc*NR,
                    β,
                    C,
                    Coffset + (i - 1)*MR*incRowC + (j - 1)*NR*incColC,
                    incRowC,
                    incColC)
            else
                gemm_micro_kernel(kc,
                    α,
                    _A,
                    (i - 1)*kc*MR,
                    _B,
                    (j - 1)*kc*NR,
                    0.0,
                    C,
                    Coffset,
                    1,
                    MR)
                gescal(mr, nr, β, C, Coffset + (i - 1)*MR*incRowC + (j - 1)*NR*incColC, incRowC, incColC)
                geaxpy(mr, nr, 1.0, _C, 1, MR, C, Coffset + (i - 1)*MR*incRowC + (j - 1)*NR*incColC, incRowC, incColC)
            end
        end
    end
    return C
end

function gemm_nn(m::Integer,
                 n::Integer,
                 k::Integer,
                 α::Float64,
                 A::Matrix{Float64},
                 incRowA::Integer,
                 incColA::Integer,
                 B::Matrix{Float64},
                 incRowB::Integer,
                 incColB::Integer,
                 β::Float64,
                 C::Matrix{Float64},
                 incRowC::Integer,
                 incColC::Integer)

    mb = div(m + MC - 1, MC)
    nb = div(n + NC - 1, NC)
    kb = div(k + KC - 1, KC)

    _mc = m % MC
    _nc = n % NC
    _kc = k % KC

    if α == 0.0 || k == 0
        gescal(m, n, β, C, incRowC, incColC)
        return C
    end

    for j = 1:nb
        nc = (j != nb || _nc == 0) ? NC : _nc

        for l = 1:kb
            kc = (l != kb || _kc == 0) ? KC : _kc
            _β = l == 1 ? β : 1.0

            pack_B(kc, nc, B, (l - 1)*KC*incRowB + (j - 1)*NC*incColB, incRowB, incColB, _B)

            for i = 1:mb
                mc = (i != mb || _mc == 0) ? MC : _mc

                pack_A(mc, kc, A, (i - 1)*MC*incRowA + (l - 1)*KC*incColA, incRowA, incColA, _A)

                gemm_macro_kernel(mc, nc, kc, α, _β, C, (i - 1)*MC*incRowC + (j - 1)*NC*incColC, incRowC, incColC)
            end
        end
    end
    return C
end

function mul!(C, A, B, α, β)
    m, ka = size(A)
    kb, n = size(B)
    @assert ka == kb && (m, n) == size(C)
    gemm_nn(m, n, ka, α,
            A, 1, m,
            B, 1, ka,
            β, C, 1, m)
end
