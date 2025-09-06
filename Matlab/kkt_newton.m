function mode1_ocp_newton()
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Single-mode nonlinear OCP via Riccati-Newton (iLQR)
    % Model:
    %   x1' = x2
    %   x2' = -1.1*x2 - 0.2*x2*|x2| + u
    % Cost:
    %   l = 0.5*(x-xref)' Q (x-xref) + u^2,  Q = diag([1,2])
    %   Vf= 0.5*(x-xref)' Q (x-xref)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    % Problem setup
    nx = 2; nu = 1;
    N  = 40;
    dt = 0.1;
    xref  = [1; 1];
    x_init = [0; 0];

    % Weights
    Q  = diag([10, 0]);   % state weight (stage)
    Qf = Q;              % terminal weight

    % Trajectories (initialize)
    X   = zeros(nx, N+1);  X(:,1) = x_init;
    U   = zeros(nu, N);
    LAM = zeros(nx, N+1);

    % initial rollout with u=0
    for i = 1:N
        [f, ~, ~, ~, ~] = f_and_derivs_mode1(X(:,i), U(:,i));
        X(:,i+1) = X(:,i) + dt*f;
    end

    % Newton settings
    max_iter = 100;
    alpha    = 0.1;   % line search factor
    regGi    = 1e-10; % control Hessian regularization
    regPmin  = 1e-12; % Riccati PD 보정

    fprintf('Start: J=%.6f\n', total_cost(X,U,xref,Q,Qf));
    
    tic;
    for it = 1:max_iter
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Linearization & KKT blocks build
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        A = cell(N,1); B = cell(N,1);
        Qxx = cell(N,1); Qxu = cell(N,1); Quu = cell(N,1);
        xbar = cell(N,1); lbar_x = cell(N,1); lbar_u = cell(N,1);

        QxxN    = Qf;                                 % terminal Hessian
        lbar_xN = gradVf(X(:,end), xref, Qf) - LAM(:,end);

        for i = 1:N
            xi = X(:,i);
            ui = U(:,i);
            lam_ip1 = LAM(:,i+1);

            % dynamics + derivatives (NEW MODEL)
            [f, dfdx, dfdu, Hxx_dyn, Hxu_dyn] = f_and_derivs_mode1(xi, ui);

            % cost (weighted)
            [~, lx, lu] = stage_cost_and_grads(xi, ui, xref, Q);

            % Hamiltonian gradient
            Hx = lx + dfdx.'*lam_ip1;
            Hu = lu + dfdu.'*lam_ip1;

            % Hamiltonian Hessian blocks (weighted)
            Hxx = Q + Hxx_dyn(lam_ip1);
            Hxu = Hxu_dyn(lam_ip1);
            Huu = 2;

            % discrete linearization (Euler)
            A{i}    = eye(nx) + dt*dfdx;
            B{i}    = dt*dfdu;
            xbar{i} = xi + dt*f - X(:,i+1);      % dyn residual

            % quadratic blocks
            Qxx{i}  = dt*Hxx;
            Qxu{i}  = dt*Hxu;
            Quu{i}  = dt*Huu;

            % residuals (stationarity)
            lbar_x{i} = dt*Hx + LAM(:,i+1) - LAM(:,i);
            lbar_u{i} = dt*Hu;
        end

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Backward Riccati recursion
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        P = cell(N+1,1); s = cell(N+1,1);
        Kmat = cell(N,1); kff = cell(N,1);

        P{N+1} = QxxN;
        s{N+1} = -lbar_xN;

        for i = N:-1:1
            Ai = A{i}; Bi = B{i};
            Qxxi = Qxx{i}; Qxui = Qxu{i}; Quui = Quu{i};
            lxi = lbar_x{i}; lui = lbar_u{i};
            Pi1 = P{i+1}; si1 = s{i+1};
            xbi = xbar{i};

            % block assembly
            Fi = Qxxi + Ai.'*Pi1*Ai;
            Hi = Qxui + Ai.'*Pi1*Bi;      % (nx x 1)
            Gi = Quui + Bi.'*Pi1*Bi;      % (1 x 1)

            % control law Δu = K Δx + kff
            Gi_reg = max(Gi, regGi);
            Kmat{i} = -(Gi_reg \ (Hi.')); % (1 x nx)
            kff{i}  = -(Gi_reg \ (Bi.'*(Pi1*xbi - si1) + lui));  % (1 x 1)

            % Riccati update
            P{i} = Fi - (Kmat{i}.' * Gi * Kmat{i});
            % PD & symmetry guard
            P{i} = 0.5*(P{i}+P{i}.');
            mEv  = min(real(eig(P{i})));
            if mEv < regPmin, P{i} = P{i} + (regPmin - mEv)*eye(nx); end

            s{i} = -( lxi + Ai.'*(Pi1*xbi - si1) + Hi*kff{i} );
        end

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Forward recursion: compute Newton step and update
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        dX = zeros(nx,N+1); dU = zeros(nu,N); dLAM = zeros(nx,N+1);
        dX(:,1) = x_init - X(:,1);

        for i = 1:N
            dU(:,i)   = Kmat{i}*dX(:,i) + kff{i};
            dLAM(:,i) = P{i}*dX(:,i) - s{i};
            dX(:,i+1) = A{i}*dX(:,i) + B{i}*dU(:,i) + xbar{i};
        end
        dLAM(:,end) = P{N+1}*dX(:,end) - s{N+1};

        % update
        X   = X   + alpha*dX;
        U   = U   + alpha*dU;
        LAM = LAM + alpha*dLAM;

        % re-rollout X with new U
        X(:,1) = x_init;
        for i = 1:N
            [f, ~, ~, ~, ~] = f_and_derivs_mode1(X(:,i), U(:,i));
            X(:,i+1) = X(:,i) + dt*f;
        end

        % report
        dXmax = max(vecnorm(dX,2,1)); dUmax = max(abs(dU));
        J = total_cost(X,U,xref,Q,Qf);
        % fprintf('Iter %3d: J=%.6f | max||dX||=%.2e, max|dU|=%.2e\n', it, J, dXmax, dUmax);
    end

    toc;

    % plots
    tgrid = 0:dt:dt*N;
    e = X - xref;                           % error over time
    state_cost = 0.5*(Q(1,1)*e(1,:).^2 + Q(2,2)*e(2,:).^2);
    figure;
    subplot(3,1,1); plot(tgrid, X(1,:),'-', tgrid, X(2,:),'-'); grid on; legend x1 x2; title('States (new model, weighted)');
    subplot(3,1,2); stairs(tgrid(1:end-1), U,'-'); grid on; title('Control u');
    subplot(3,1,3); plot(tgrid, state_cost,'-'); grid on; title('0.5 (x-x_{ref})^T Q (x-x_{ref})');
end

% ===================== helpers (NEW MODEL + WEIGHTS) =====================

function g = gradVf(x, xr, Qf)
    g = Qf*(x - xr);
end

function [l, lx, lu] = stage_cost_and_grads(x, u, xr, Q)
    e  = (x - xr);
    lx = Q*e;           % ∂l/∂x
    lu = 2*u;           % ∂l/∂u
    l  = 0.5*(e.'*Q*e) + u^2;
end

function [f, dfdx, dfdu, Hxx_dyn, Hxu_dyn] = f_and_derivs_mode1(x, u)
    % Model:
    % x1' = x2
    % x2' = -1.1*x2 - 0.2*x2*|x2| + u
    x2 = x(2);

    f1 = x2;
    f2 = -1.1*x2 - 0.2*x2*abs(x2) + u;
    f  = [f1; f2];

    % Jacobians
    dfdx = [ 0,                    1;
             0,  -1.1 - 0.4*abs(x2) ];
    dfdu = [0; 1];

    % Dynamics curvature for Hamiltonian (second derivatives wrt x)
    sgn = sign(x2); % sign(0)=0
    Hxx_dyn = @(lam) lam(2)*[0,0;0, -0.4*sgn];  % only f2_x2x2 nonzero
    Hxu_dyn = @(lam) [0; 0];                    % mixed x-u second derivatives = 0
end

function J = total_cost(X,U,xr,Q,Qf)
    % accumulates weighted stage + terminal
    J = 0;
    Nloc = size(U,2);
    for ii=1:Nloc
        e = X(:,ii)-xr;
        J = J + 0.5*(e.'*Q*e) + U(:,ii)'*U(:,ii);
    end
    eN = X(:,end)-xr;
    J = J + 0.5*(eN.'*Qf*eN);
end
