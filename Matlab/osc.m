function switched_algo2_two_modes_dwellbarrier()
%=============================================================
% Riccati-Newton (Algorithm 2) for a 2-mode switched OCP (1 -> 2)
% Model 1:
%   x1' = x2
%   x2' = -1.1*x2 - 0.2*x2*|x2| + u
% Model 2:
%   x1' = x2
%   x2' = -0.5*x2 - 0.05*x2*|x2| + u
%
% Cost:
%   l = 0.5*(x-xref)' Q (x-xref) + u^2  - mu_u[log(umax-u)+log(u-umin)]
%   Vf= 0.5*(x-xref)' Q (x-xref)
% with Q = diag([0,100])
%
% Discretization: Δτ_k = (t_k - t_{k-1})/N_k, k=1,2  (N=N1+N2)
%
% Implements Section 4 + Algorithm 2 + barriers:
%  - Input bounds by log barrier on u (per-stage: value/grad/hess)
%  - Dwell-time bounds by log barrier on t1 (boundary terms in (21)(22))
%  - Fraction-to-boundary line search keeps u strictly interior
%=============================================================

% ---------- Problem setup ----------
nx = 2; nu = 1;
N1 = 20; N2 = 20;  N = N1+N2;  K = 1;

Q  = diag([0,100]);   Qf = Q;
xref = [1;2];  x_init = [0;0];

T = 4.0; t0 = 0.0; tf = T;

% Input bounds (interior point)
umin = -1.0; umax = 1.0;
mu_u0 = 1e-2;       % initial u-barrier weight
mu_u_min = 1e-6;    % floor
decay_u = 0.95;     % schedule
eps_int = 1e-12;    % interior slack for u
gamma_ftb = 0.995;  % fraction-to-boundary

% Dwell-time barrier parameters
tau1_min = 0.3;     % min dwell in phase 1
tau2_min = 1.0;     % min dwell in phase 2
mu_tau   = 1e-2;    % dwell barrier strength
eps_t    = 1e-9;

% Phase index ranges
Ik_start = [1, 1+N1];
Ik_end   = [N1, N];
Nk       = [N1, N2];

% Initial switching time (respect dwell lower bounds)
tk = [ max(t0+tau1_min+eps_t, t0 + (tf-t0)*N1/N), tf ];

% Trajectories
X   = zeros(nx, N+1);  X(:,1) = x_init;
U   = zeros(nu, N);    % start at 0 (interior)
LAM = zeros(nx, N+1);

% Initial rollout with U=0
X = rollout_forward(X,U,tk,Ik_start,Ik_end);

% ---------- Newton / Riccati options ----------
max_iter = 50;
alpha_base = 0.2;    % base step for (Δx,Δu,Δλ,Δt)
regGi    = 1e-12;
regPmin  = 1e-12;
ts_max   = 0.5;

fprintf('Start: J=%.6f | t1=%.4f\n', total_cost_with_barriers(X,U,xref,Q,Qf,umin,umax,mu_u0,tk,t0,tf,tau1_min,tau2_min,mu_tau), tk(1));

for it = 1:max_iter
    % barrier weight for u this iteration
    mu_u = max(mu_u_min, mu_u0 * decay_u^(it-1));

    % per-phase Δτ_k
    dtau1 = (tk(1)-t0)/N1;
    dtau2 = (tk(2)-tk(1))/N2;
    dtaus = [dtau1, dtau2];

    % ---------- Build KKT blocks per stage ----------
    A=cell(N,1); B=cell(N,1);
    Qxx=cell(N,1); Qxu=cell(N,1); Quu=cell(N,1);
    xbar=cell(N,1); lbar_x=cell(N,1); lbar_u=cell(N,1);
    hx=cell(N,1); hu=cell(N,1); a=cell(N,1); Hval=zeros(N,1);

    QxxN = Qf;
    lbar_xN = gradVf(X(:,end), xref, Qf) - LAM(:,end);

    for i=1:N
        k  = (i<=N1) + 1*(i>N1);      % 1 or 2
        Dt = dtaus(k);
        Nk_k = Nk(k);

        xi = X(:,i);  ui = U(:,i);  lam_ip1 = LAM(:,i+1);

        % dynamics + derivatives
        [f, dfdx, dfdu, Hxx_dyn, Hxu_dyn] = f_and_derivs_mode(k, xi, ui);

        % stage cost + u-barrier
        [l, lx, lu, huu_bar] = stage_cost_and_grads_with_ubarrier(xi, ui, xref, Q, mu_u, umin, umax);

        % Hamiltonian and derivatives
        H  = l + lam_ip1.'*f;
        Hx = lx + dfdx.'*lam_ip1;
        Hu = lu + dfdu.'*lam_ip1;

        % Hessians
        Hxx = Q + Hxx_dyn(lam_ip1);
        Hxu = Hxu_dyn(lam_ip1);
        Huu = 2 + huu_bar;     % u^2 term + barrier curvature

        % Discrete linearization
        A{i}    = eye(nx) + Dt*dfdx;
        B{i}    = Dt*dfdu;
        xbar{i} = xi + Dt*f - X(:,i+1);

        % Quadratic blocks
        Qxx{i}  = Dt*Hxx;
        Qxu{i}  = Dt*Hxu;
        Quu{i}  = Dt*Huu;

        % time-sensitivity vectors
        hx{i}   = (1/Nk_k)*Hx;
        hu{i}   = (1/Nk_k)*Hu;       % includes barrier gradient part in Hu
        a{i}    = (1/Nk_k)*f;
        Hval(i) = (1/Nk_k)*H;        % includes barrier value

        % residuals
        lbar_x{i} = Dt*Hx + LAM(:,i+1) - LAM(:,i);
        lbar_u{i} = Dt*Hu;
    end

    % ---------- Backward recursion ----------
    P=cell(N+1,1); s=cell(N+1,1);
    Kmat=cell(N,1); kff=cell(N,1);
    Tcoef=cell(N,1); Wcoef=cell(N,1);
    iota_vec=cell(N+1,1); ihat_vec=cell(N+1,1);

    xi_s=zeros(N+1,1); chi_s=zeros(N+1,1);
    rho_s=zeros(N+1,1); eta_s=zeros(N+1,1); iota_s=zeros(N+1,1);

    P{N+1}=QxxN; s{N+1}=-lbar_xN;
    iota_vec{N+1}=zeros(nx,1); ihat_vec{N+1}=zeros(nx,1);

    sigma_phase=zeros(K,1);
    vm_at_ik=cell(K,1);
    xi_at_ik=zeros(K,1); chi_at_ik=zeros(K,1);
    rho_at_ik=zeros(K,1); eta_at_ik=zeros(K,1);
    iota_s_at_ik=zeros(K,1);
    eta_iota_extra=cell(K,1);  % dwell gradient term

    for k=2:-1:1
        for i=Ik_end(k):-1:Ik_start(k)
            Ai=A{i}; Bi=B{i};
            Qxxi=Qxx{i}; Qxui=Qxu{i}; Quui=Quu{i};
            lxi=lbar_x{i}; lui=lbar_u{i};
            axi=a{i}; hxi=hx{i}; hui=hu{i};
            Pi1=P{i+1}; si1=s{i+1};
            iv1=iota_vec{i+1}; ih1=ihat_vec{i+1};
            xbi=xbar{i};

            % (15)
            Fi = Qxxi + Ai.'*Pi1*Ai;
            Hi = Qxui + Ai.'*Pi1*Bi;
            Gi = Quui + Bi.'*Pi1*Bi;

            % ψ, φ
            psi_x = hxi + Ai.'*Pi1*axi + Ai.'*iv1;
            psi_u = hui + Bi.'*Pi1*axi + Bi.'*iv1;
            phi_x = Ai.'*ih1;
            phi_u = Bi.'*ih1;

            % (16)
            Gi_reg = max(Gi, regGi);
            Kmat{i}  = -(Gi_reg \ (Hi.'));
            kff{i}   = -(Gi_reg \ (Bi.'*(Pi1*xbi - si1) + lui));
            Tcoef{i} = -(Gi_reg \  psi_u);
            Wcoef{i} = -(Gi_reg \  phi_u);

            % (17)
            P{i} = Fi - (Kmat{i}.' * Gi * Kmat{i});
            P{i} = 0.5*(P{i}+P{i}.');
            mEv = min(real(eig(P{i})));
            if mEv < regPmin, P{i} = P{i} + (regPmin - mEv)*eye(nx); end

            s{i}        = -( lxi + Ai.'*(Pi1*xbi - si1) + Hi*kff{i} );
            iota_vec{i} =  psi_x + (Kmat{i}.' * psi_u);
            ihat_vec{i} =  phi_x + (Kmat{i}.' * phi_u);

            % (19)
            xi_s(i)   = xi_s(i+1)   + axi.'*(Pi1*axi + 2*iv1) + psi_u.'*Tcoef{i};
            eta_s(i)  = eta_s(i+1)  + Hval(i) + axi.'*(Pi1*xbi - si1) + iv1.'*xbi + psi_u.'*kff{i};
            chi_s(i)  = chi_s(i+1)  + iv1.'*axi + psi_u.'*Wcoef{i};
            rho_s(i)  = rho_s(i+1)  + (phi_u.'*Wcoef{i});
            iota_s(i) = iota_s(i+1) + (ih1.'*xbi + phi_u.'*kff{i});
        end

        % boundary k=2 → i1 = N1+1
        if k>=2
            ik = Ik_start(k);

            % base sigma (22)
            sigma_raw = xi_s(ik) - 2*chi_s(ik) + rho_s(ik);

            % dwell barrier add
            tau1 = tk(1) - t0;  tau2 = tf - tk(1);
            phi_grad = -mu_tau*( 1/max(tau1 - tau1_min,eps_t) - 1/max(tau2 - tau2_min,eps_t) );
            phi_hess =  mu_tau*( 1/max(tau1 - tau1_min,eps_t)^2 + 1/max(tau2 - tau2_min,eps_t)^2 );

            sigma_raw = sigma_raw + phi_hess;
            eta_minus_iota = (eta_s(ik) - iota_s(ik)) + phi_grad;

            sig = modify_sigma(sigma_raw, eta_minus_iota, ts_max);

            sigma_phase(k-1)  = sig;
            vm_at_ik{k-1}     = (iota_vec{ik} - ihat_vec{ik});
            xi_at_ik(k-1)     = xi_s(ik);   chi_at_ik(k-1) = chi_s(ik);
            rho_at_ik(k-1)    = rho_s(ik);  eta_at_ik(k-1) = eta_s(ik);
            iota_s_at_ik(k-1) = iota_s(ik);
            eta_iota_extra{k-1} = phi_grad; % store dwell grad
        end

        % (23) tilde updates
        if k>=2
            ik = Ik_start(k);
            Ptilde   = P{ik};
            s_tilde  = s{ik}  + ((eta_s(ik)-iota_s(ik))/sigma_phase(k-1)) * (iota_vec{ik}-ihat_vec{ik});
            ihat_tld =  iota_vec{ik} - ((xi_s(ik)-chi_s(ik))/sigma_phase(k-1)) * (iota_vec{ik}-ihat_vec{ik});

            rho_tld    = xi_s(ik) - ((xi_s(ik)-chi_s(ik))^2)/sigma_phase(k-1);
            iota_s_tld = eta_s(ik) - ((xi_s(ik)-chi_s(ik))*(eta_s(ik)-iota_s(ik)))/sigma_phase(k-1);

            P{ik}        = Ptilde;
            s{ik}        = s_tilde;
            iota_vec{ik} = zeros(nx,1);
            ihat_vec{ik} = ihat_tld;
            xi_s(ik)=0; chi_s(ik)=0; eta_s(ik)=0;
            rho_s(ik)=rho_tld; iota_s(ik)=iota_s_tld;
        end
    end

    % ---------- Forward recursion ----------
    dX=zeros(nx,N+1); dU=zeros(nu,N); dLAM=zeros(nx,N+1);
    dtk=zeros(K,1);

    dX(:,1) = x_init - X(:,1);

    % Δt1 via (21)
    i1 = Ik_start(1);
    v1 = vm_at_ik{1}; sig1 = sigma_phase(1);
    num1 = v1.'*dX(:,i1) + (xi_at_ik(1)-chi_at_ik(1))*0 ...
         + (eta_at_ik(1)-iota_s_at_ik(1)) + eta_iota_extra{1};
    dtk(1) = -(1/sig1) * num1;

    % Phase 1
    for i=Ik_start(1):Ik_end(1)
        dt_prev = 0; dt_curr = dtk(1);
        dU(:,i)   = Kmat{i}*dX(:,i) + kff{i} + Tcoef{i}*(dt_curr - dt_prev) + Wcoef{i}*(-dt_curr);
        dLAM(:,i) = P{i}*dX(:,i) - s{i} + iota_vec{i}*(dt_curr - dt_prev) + ihat_vec{i}*(-dt_curr);
        dX(:,i+1) = A{i}*dX(:,i) + B{i}*dU(:,i) + a{i}*(dt_curr - dt_prev) + xbar{i};
    end
    % Phase 2
    for i=Ik_start(2):Ik_end(2)
        dt_prev = dtk(1); dt_curr = 0;
        dU(:,i)   = Kmat{i}*dX(:,i) + kff{i} + Tcoef{i}*(dt_curr - dt_prev) + Wcoef{i}*(-dt_curr);
        dLAM(:,i) = P{i}*dX(:,i) - s{i} + iota_vec{i}*(dt_curr - dt_prev) + ihat_vec{i}*(-dt_curr);
        dX(:,i+1) = A{i}*dX(:,i) + B{i}*dU(:,i) + a{i}*(dt_curr - dt_prev) + xbar{i};
    end
    dLAM(:,end) = P{N+1}*dX(:,end) - s{N+1};

    % ---------- Fraction-to-boundary line search for u ----------
    alpha_max = 1.0;
    for i=1:N
        ui = U(:,i); dui = dU(:,i);
        if dui > 0
            alpha_max = min(alpha_max, (umax - eps_int - ui)/dui);
        elseif dui < 0
            alpha_max = min(alpha_max, (ui - (umin + eps_int))/(-dui));
        end
    end
    alpha_eff = min(alpha_base, gamma_ftb * max(0, alpha_max));

    % ---------- Update ----------
    X   = X   + alpha_eff*dX;
    U   = U   + alpha_eff*dU;
    LAM = LAM + alpha_eff*dLAM;
    tk(1) = tk(1) + alpha_eff*dtk(1);

    % dwell hard guard
    tk(1) = min( max(tk(1), t0 + tau1_min + eps_t), tf - tau2_min - eps_t );

    % rollout with updated times and U
    X = rollout_forward(X,U,tk,Ik_start,Ik_end);

    % report (cost includes u-barrier + dwell-barrier)
    J = total_cost_with_barriers(X,U,xref,Q,Qf,umin,umax,mu_u,tk,t0,tf,tau1_min,tau2_min,mu_tau);
    fprintf('Iter %2d: J=%.6f | t1=%.4f | alpha=%.3f | max|dU|=%.2e\n', it, J, tk(1), alpha_eff, max(abs(dU)));
end

% ----- plots -----
[tgrid, phase_id] = time_grid_from_tk(N1,N2,tk,t0,tf);
figure;
subplot(3,1,1); plot(tgrid, X(1,:),'-', tgrid, X(2,:),'-'); grid on; legend x1 x2; title('States');
subplot(3,1,2); stairs(tgrid(1:end-1), U,'-'); grid on; ylim([umin umax]); title('u (interior w/ barrier)');
subplot(3,1,3); stairs(tgrid(1:end-1), phase_id, '-'); ylim([0.5 2.5]); yticks([1 2]); yticklabels({'mode1','mode2'}); grid on; title('Active mode');

end % main

%================= helpers =================

function X = rollout_forward(X,U,tk,Ik_start,Ik_end)
    N1 = Ik_end(1)-Ik_start(1)+1;
    N2 = Ik_end(2)-Ik_start(2)+1;
    dt1 = (tk(1)-0.0)/N1;
    dt2 = (tk(2)-tk(1))/N2;
    for i=1:Ik_end(1)
        [f,~,~,~,~] = f_and_derivs_mode(1, X(:,i), U(:,i));
        X(:,i+1) = X(:,i) + dt1*f;
    end
    for i=Ik_start(2):Ik_end(2)
        [f,~,~,~,~] = f_and_derivs_mode(2, X(:,i), U(:,i));
        X(:,i+1) = X(:,i) + dt2*f;
    end
end

function [tgrid, phase_id] = time_grid_from_tk(N1,N2,tk,t0,tf)
    dt1=(tk(1)-t0)/N1; dt2=(tf-tk(1))/N2;
    tgrid=[t0:dt1:tk(1), tk(1)+dt2:dt2:tf]; %#ok<NBRAK>
    if numel(tgrid)~=N1+N2+1, tgrid = linspace(t0, tf, N1+N2+1); end
    phase_id = [ones(1,N1), 2*ones(1,N2)];
end

function [l, lx, lu, huu_bar] = stage_cost_and_grads_with_ubarrier(x, u, xr, Q, mu_u, umin, umax)
    % base quadratic
    e  = (x - xr);
    lq = 0.5*(e.'*Q*e) + u^2;
    lx = Q*e;
    lu_quad = 2*u;

    % u log-barrier
    [bval, bgrad, bhess] = barrier_terms_u(u, umin, umax, mu_u);

    % totals
    l       = lq + bval;
    lu      = lu_quad + bgrad;
    huu_bar = bhess;     % to be added to Huu
end

function [bval, bgrad, bhess] = barrier_terms_u(u, umin, umax, mu)
    % -mu[ log(umax-u) + log(u-umin) ], interior only
    bval  = -mu*( log(umax - u) + log(u - umin) );
    bgrad =  mu*( 1/(umax - u) - 1/(u - umin) );
    bhess =  mu*( 1/(umax - u)^2 + 1/(u - umin)^2 );
end

function g = gradVf(x, xr, Qf)
    g = Qf*(x - xr);
end

function [f, dfdx, dfdu, Hxx_dyn, Hxu_dyn] = f_and_derivs_mode(mode, x, u)
    x2 = x(2);
    switch mode
        case 1
            f2 = -1.1*x2 - 0.2*x2*abs(x2) + u;
            d_damp = -1.1; c = 0.2;
        case 2
            f2 = -0.5*x2 - 0.05*x2*abs(x2) + u;
            d_damp = -0.5; c = 0.05;
    end
    f = [x(2); f2];

    dfdx = [0, 1;
            0, d_damp - 2*c*abs(x2)];   % d/dx2 of -c x2|x2| = -2c|x2|
    dfdu = [0; 1];

    sgn = sign(x2);
    Hxx_dyn = @(lam) lam(2)*[0,0;0, -2*c*sgn];   % only f2_x2x2 ≠ 0
    Hxu_dyn = @(lam) [0; 0];
end

function sgm = modify_sigma(sgm_raw, eta_minus_iota_s, ts_max_local)
    smin = abs(eta_minus_iota_s) / max(ts_max_local,1e-6);
    if sgm_raw > smin, sgm = sgm_raw;
    else,              sgm = abs(sgm_raw) + smin;
    end
end

function J = total_cost_with_barriers(X,U,xr,Q,Qf,umin,umax,mu_u,tk,t0,tf,tau1_min,tau2_min,mu_tau)
    % stage + terminal + u-barrier + dwell-barrier (for reporting)
    J=0; Nloc=size(U,2);
    for ii=1:Nloc
        e=X(:,ii)-xr;
        u=U(:,ii);
        J = J + 0.5*(e.'*Q*e) + u'*u ...
              - mu_u*( log(umax - u) + log(u - umin) );
    end
    eN=X(:,end)-xr; J = J + 0.5*(eN.'*Qf*eN);

    % dwell barrier value
    tau1 = tk(1)-t0; tau2 = tf - tk(1);
    J = J - mu_tau*( log(tau1 - tau1_min) + log(tau2 - tau2_min) );
end
