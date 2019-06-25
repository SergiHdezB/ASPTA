clear variables

%CONSTANTS
L = 3; %Channel delay
D = 100; %Number of inputs
N = 1; %Number of observations
M = 2; %Receptors
N_ESTIMATIONS = 100; %Number of channel estimations
N_ITERATIONS = 1; %Number of iterations for channel estimation
F = [eye(D) zeros(D,L)]; %[I|O]
J = diag(ones((D+L)-1,1),1); %Shift matrix

%Model Generation
decay = zeros(L+1,1); %Delay channel decay
w = 2; %Channel decay weights
mu_h = 0; %Channel mean
sigma_h = 1; %Channel variance
mu_s = 0; %Input mean
sigma_s = 1; %Input variance
Rs = eye(D+L); %Input S covariance
mu_v = 0; %Mean variance
sigma_v = 0.01; %Noise variance


%Signal input
input = normrnd(mu_s, sigma_s, [1,D]);
S = zeros(D+L,N);
S(1,1:N) = input(1:N);

next_row = circshift(input,1);
for m = 2:(D+L)
    S(m,1:N) = next_row(1,1:N);
    next_row = circshift(next_row,1);
end

S = triu(S);
s = S(1,:); %Simbols

estimations = 1;
errors = zeros(N_ESTIMATIONS,1);
while(estimations <= N_ESTIMATIONS)
    
    %Estimation parameters
c = 1; %Precision noise (v) gamma pdf parameter
d = 1; %Precision noise (v) gamma pdf parameter
beta_expectation = 1;
H_expectationMdim = zeros(D,D+L);
h_expectation = [1 0 0 -1; 0 0 0 1];
h_variance = ones(L+1);
s_expectation = zeros(D+L,M,N);
s_variance = eye(D+L);

%Channel
h = zeros(M,L);
for l = 0:L
    decay(l+1) = exp(-w*l);
    h(:,l+1) = normrnd(mu_h,decay(l+1)*sigma_h,[M,1]);
end

H = zeros(M*D, (D+L));
H_Mdim = zeros(D,D+L,M);
for k = 0:floor((M*D-1)/M)
    m = M*k + 1;
    H(m:(m+M-1),:) = [zeros(M,k) h zeros(M, D-k-1)];
    for n = 1:M
        H_Mdim(k+1,:,n) = [zeros(1,k) h(n,:) zeros(1, D-k-1)];
    end
end

X_desordenado = H*S;
X = zeros(D,M*N);
X_Mdim = zeros(D,M,N);
x = zeros(D,M);
for p = 0:(N-1)
    n = M*p + 1;
    x = zeros(D,M);
    x_desordenado = X_desordenado(:,p+1);
    for k = 0:floor((M*D-1)/M)
        m = M*k + 1;
        x(k+1,:) = transpose(x_desordenado(m:(m+M-1)));
    end
    X(:,n:(n+M-1)) = x;
    X_Mdim(:,:,p+1) = x;
end

%V noise
for n = 1:N
    X_Mdim(:,:,n) = X_Mdim(:,:,n) + normrnd(mu_v, sigma_v, [D,M]);
end


%%TEST ---------------------------------------------------------------  

iterations = 1;
while(iterations <= N_ITERATIONS)
%   q(S) parameters

%H expectation and variance
H_expectation = zeros(D*M,D+L);
for l = 0:L   
    H_expectation = kron(F*J^l, h_expectation(:,l+1)) + H_expectation;
end


for k = 0:floor((M*D-1)/M)
    m = M*k + 1;
    for n = 1:M
        H_expectationMdim(k+1,:,n) = [zeros(1,k) H_expectation(n,1:(L+1)) zeros(1, D-k-1)];
    end
end

H_variance = eye(D+L);
for i = 0:(L-1)
    for j = 0:(L-1)         
        H_variance = ((J^i)')*(F')*F*(J^j)*h_variance(i+1,j+1)...
                     + H_variance;
    end
end

Ps = beta_expectation*H_variance + Rs^-1;

for n = 1:N
    for m = 1:M
        s_expectation(:,m,n) = (Ps^-1)*beta_expectation*H_expectationMdim(:,:,m)'*X_Mdim(:,m,n); 
        s_variance(:,:,n) = (Ps^-1) + s_expectation(:,:,n)*s_expectation(:,:,n)';
    end
end

% %   q(h) parameters

aux_y = zeros(M,1);
y = zeros((L+1)*M,1);
for l = 0:L
    m = M*l + 1;
    for n = 1:N
        for d = 0:(D-1)
            if(n > d && n > (l+d))
                aux_y = (s_expectation(1,:,(n-l-d))').*(X_Mdim(1,:,n-d)') + aux_y;
            end                
        end
    end
    y(m:(m+M-1)) = aux_y;
end

gamma = zeros(L+1);
for i = 0:L
    for j = 0:L
        for n = 1:N
          gamma(i+1, j+1) = trace(((J^i)')*(F')*F*(J^j)*s_variance(:,:,n)) + gamma(i+1, j+1);
        end
    end    
end

Ph_L = (beta_expectation*gamma+diag(decay));
Ph = kron(Ph_L,eye(M));
h_expectation = beta_expectation*(Ph^-1)*y;
h_expectation = reshape(h_expectation,L+1,M)';
h_variance = (Ph_L^-1) + (h_expectation')*h_expectation;


% %   q(B) parameters

d_prima_matrix = zeros(M);
for n = 1:N
    for m = 1:M        
    d_prima_matrix = (X_Mdim(:,m,n)')*X_Mdim(:,m,n) - 2*(X_Mdim(:,m,n)')*H_expectationMdim(:,:,m)*s_expectation(:,m,n)...
                + trace(s_variance(:,:,n)*H_variance) + d_prima_matrix;
    end
end

d_prima_matrix = (1/2)*d_prima_matrix;
d_prima = d + d_prima_matrix(1,1);
c_prima = c + N*M*D/2;
beta_expectation = c_prima/d_prima;

iterations = iterations +1;
end


diff = h_expectation - h;
distance = norm(diff);
h_energia = norm(h);
errors(estimations) = distance/norm(h);
estimations = estimations + 1;

end

NMSE = (1/N_ESTIMATIONS)*sum(errors);



    
