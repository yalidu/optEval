%%
% Test matlab code for c-version of OptSpace with interface using files.

function testing

    clear all
    seed = 2009;

    m = 1000;  
    n = 1000;
    r = 5;
    eps = 50;
    SR = eps/sqrt(m*n); 
    p = floor(SR*m*n); 
    FR = (r*(m+n-r)/p);
    if (FR > 1); error('FR > 1'); end
    rand('state',seed);  
    randn('state',seed);
    Idx = randperm(m*n); 
    Idx = Idx(1:p); 
    Idx = sort(Idx);
    U = randn(m,r); 
    V = randn(n,r); 
    M = U*V';
    normM = norm(M, 'fro'); 
    b = M(Idx); 
    ME = sparse(m,n); 
    ME(Idx) = b;

%% Run OptSpace without noise
    tol = 1e-4;
    estimated_rank = r; % We can use any estimation method to estimate the rank here.
    niter = 500;
    % Write sample matrix ME to a file 'input'. 
    % The order has to be 'precisely' increasing column indices first 
    % and increasing row indices next.
    % For example 
    % row col value
    %   2   1   0.1
    %   10  1   0.3
    %   21  1   0.2
    %   1   2   0.4
    %   7   2   0.5
    %   8   3   0.6
    % ...
    input_file='input';
    fid = fopen(input_file,'w');
    for col=1:n
        for row=1:m
            if (ME(row,col)~=0)
                fprintf(fid,'%i\t %i\t %e\n',row,col,ME(row,col));
            end
        end
    end
    fclose(fid);

    % Run OptSpace C-version, 'tsolve' is the computation time of
    % OptSpace after reading ME from file to completing OptSpace
    command=['./test ' num2str(m) ' ' num2str(n) ' ' num2str(nnz(ME)) ' ' num2str(estimated_rank) ' ' input_file ' ' num2str(niter) ' ' num2str(tol) ];
    [status result] = unix(command,'-echo');
    tsolve_nonoise = result(end-11:end-2);  
    
    % Compute RMSE, by reading X,S,Y from 'outputX' 'outputS' 'outputY'
    fid = fopen('outputX', 'r');
    X = fscanf(fid, '%e', [estimated_rank m]);
    X = X';
    fclose(fid);
    fid = fopen('outputY', 'r');
    Y = fscanf(fid, '%e', [estimated_rank n]);
    Y = Y';
    fclose(fid);
    fid = fopen('outputS', 'r');
    S = fscanf(fid, '%e', [estimated_rank estimated_rank]);
    S = S';
    fclose(fid);
    err_nonoise_absolute = norm(X*S*Y'-M,'fro')/sqrt(m*n);
    %err_nonoise_relative = norm(X*S*Y'-M,'fro')/normM;

%% Run OptSpace with noise
    sig = 0.01; % noise std
    N   = M + sig*randn(m,n);
    b = N(Idx); 
    ME = sparse(m,n); 
    ME(Idx) = b;

    tol = 1e-4;
    estimated_rank = r; % We can use any estimation method to estimate the rank here.
    niter = 50;
    % Write sample matrix ME to a file 'input'. 
    % The order has to be 'precisely' increasing column indices first 
    % and increasing row indices next.
    % For example 
    % row col value
    %   2   1   0.1
    %   10  1   0.3
    %   21  1   0.2
    %   1   2   0.4
    %   7   2   0.5
    %   8   3   0.6
    % ...
    input_file='input';
    fid = fopen(input_file,'w');
    for col=1:n
        for row=1:m
            if (ME(row,col)~=0)
                fprintf(fid,'%i\t %i\t %e\n',row,col,ME(row,col));
            end
        end
    end
    fclose(fid);

    % Run OptSpace C-version, 'tsolve' is the computation time of
    % OptSpace after reading ME from file to completing OptSpace
    command=['./test ' num2str(m) ' ' num2str(n) ' ' num2str(nnz(ME)) ' ' num2str(estimated_rank) ' ' input_file ' ' num2str(niter) ' ' num2str(tol) ];
    [status result] = unix(command,'-echo');
    tsolve_noise = result(end-11:end-2);  
    
    % Compute RMSE, by reading X,S,Y from 'outputX' 'outputS' 'outputY'
    fid = fopen('outputX', 'r');
    X = fscanf(fid, '%e', [estimated_rank m]);
    X = X';
    fclose(fid);
    fid = fopen('outputY', 'r');
    Y = fscanf(fid, '%e', [estimated_rank n]);
    Y = Y';
    fclose(fid);
    fid = fopen('outputS', 'r');
    S = fscanf(fid, '%e', [estimated_rank estimated_rank]);
    S = S';
    fclose(fid);
    err_noise_absolute = norm(X*S*Y'-M,'fro')/sqrt(m*n);
    %err_noise_relative = norm(X*S*Y'-M,'fro')/normM;
%% Print RMSE
    fprintf(1,['RMSE (without noise)   \t\t: %e in ' tsolve_nonoise ' seconds\n'],err_nonoise_absolute);
    fprintf(1,['RMSE (with noise var %.3f) \t: %e in ' tsolve_noise ' seconds\n'],sig,err_noise_absolute);
end

