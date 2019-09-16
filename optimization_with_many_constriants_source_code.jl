using PyPlot
# set parameters
const M=100
const N=100                      # dimensions
const f=1000                     # functions
const c=500                      # constraints

# generate functions
A= randn(f,M,N)                 # vector of ai
b= randn(f,M)                   # vector of bi
# generate constraints
Ac= randn(c,N)                  # vector of ai
bc= randn(c)                    # vector of bi

iterations=10^7
J= zeros(f+c,N)
Jmean= zeros(N)
τ=1
λ=1

function err(xk)                # sum of errors of all (quadratic) functions
    err=0
    for i=1:f
        err += (A[i]*xk-b[i])'*(A[i]*xk-b[i])
    end
    return err[1]
end

function computeJacobian()      # compute jacobian of (quadratic) functions
    jacobian=zeros(f,N)
    for col=1:f
        jacobian[col,:,:]=2*A[col,:,:]'*(A[col,:,:]*x-b[col,:,:])
    end
    return jacobian
end

function computeLmax()
    Lmax = 1
    for col=1:f
        e = eigmax(A[col,:,:]'*A[col,:,:])
        Lmax = max(e, Lmax)
    end
    return Lmax
end

L = computeLmax()
α = 1/(3L)
s = 1-1/(f+c)             # new jacobian
r = 1-s                   # previous jacobian
λ = 100

function setS(size,minibatch)   # get a random vector of size f with τ ones
    S=zeros(size,1)
    for i=1:minibatch
        r=rand(1:size)
        while S[r]==1
            r=rand(1:size)
        end
        S[r]=1
    end
    return S
end

function updateJacobian(x)
    S=setS(f+c,τ)
    changeJmean= zeros(N)
    for col=1:f
        if S[col]==1
            newGrad = J[col,:,:]*r + ( 2*A[col,:,:]'*(A[col,:,:]*x-b[col,:,:]) )*s
            changeJmean += newGrad - J[col,:,:]
            J[col,:,:] = newGrad
        end
    end
    for col=1:c
        if S[f+col]==1
            newGrad = J[f+col,:,:]*r + λ*(2*Ac[col,:].*(Ac[col,:]'*x - bc[col])/(Ac[col,:]⋅Ac[col,:]))*s
            changeJmean += newGrad - J[f+col,:,:]
            J[f+col,:,:] = newGrad
        end
    end
    return Jmean + changeJmean/f/τ
end

ts=[1,50,300,600,900,1200,1500,2000, f+c]
as=[0.0003, 1/(2L), 1/(3L), 0.1, 0.001, 0.0001, 0.00001, 1.0, 10.0]
lambdas=[0.1,1,10,100,1000]
# try each combination of τ, α and λ above

for init=1:5
    for lamb=1:size(lambdas,1)
        λ=lambdas[lamb]
        # initialize algorithm
        x0= randn(N)                    # N vector
        J0= zeros(f+c,N)                # vector of fi gradients
        J0mean= squeeze(mean(J0,1),1)   # precomputation of mean of jacobian

        α= 0.00003
        τ= 1
        J= J0*1
        Jmean= J0mean*1
        xopt= x0*1
        # find optimal x
        for it=1:iterations*2
            Jmean = updateJacobian(xopt)
            xopt -= α*Jmean
        end
        taus=zeros(ts)
        grads=zeros(ts)

        for t=1:size(ts,1)
            τ=ts[t]
            print("τ: ", τ, "\n")
            taus[t]=τ
            grads[t]=(f+c)*iterations
            prev_it=trunc(Int, iterations/2)
            # find try  different alphas
            for a=1:size(as,1)
                α=as[a]                  # stepsize

                dist=zeros(iterations)
                # reset variables to initial state
                x= x0*1
                J= J0*1
                Jmean= J0mean*1
                # simulate algorithm, find minimal number of iteration to converge
                for it=1:prev_it*2       # iteration after actual minimum of iterations are not interesting
                    Jmean = updateJacobian(x)
                    x -= α*Jmean

                    dist[it]=norm(xopt-x)
                    if(dist[it]<1/10^12)
                        # simulation converged
                        prev_it=min(it, prev_it)    # update minimum of iterations to converge
                        if grads[t]>it*τ            # minimum of gradients computed
                            grads[t] = it*τ
                        end
                        print("|α: ", α," converged in iteration ", it, "\n")
                        break
                    end
                end
            end
            scatter(taus, grads, label="gradient computation")
            title(string("λ=", λ))
            xlabel("minibatch size")
            legend()
            savefig(string("/home/slavo/Desktop/project/graphs/timeplot_", lamb, "_", init) )
            close()
        end
    end
end
