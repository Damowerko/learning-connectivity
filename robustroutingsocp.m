function [slack_var, routing_vars, status] = robustroutingsocp(x, qos, constrain_slack)
% ROBUSTROUTINGSOCP solve the robust routing SOCP formulation
%
% inputs:
%   x   - 2Nx1 column vector of [x;y] agent positions stacked
%   qos - a Kx1 array of structs containing:
%         flow.src   - the list of source nodes
%         flow.dest  - the list of destination nodes
%         margin     - the margin with which the constraint must be met
%         confidence - probabilistic confidence that the constraint holds
%
% outputs:
%   slack_var - the slack variable from the SOCP solution
%   status    - the status of the SOCP; if isnan(status) == true then no
%               feasible was found

N = size(x,1)/2;
K = length(qos);

%%  form SOCP coefficient matrices

R = linkratematrix(x);
[A,B] = makesocpconsts(qos,R);

%% Solve SOCP

% confidence threshold
conf = norminv(vertcat(qos(:).confidence), 0, 1);

% node margins
m_ik = zeros(N,K);
for k = 1:K
  for i = 1:length(qos(k).flow.src)
    m_ik(qos(k).flow.src(i),k) = qos(:).margin;
  end
end

% slack bound
slack_bound = 0;
if ~constrain_slack
  slack_bound = -1e10; % sufficiently large number to be unbounded
end

cvx_begin quiet
  variables routing_vars(N,N,K) slack_var
  y = [routing_vars(:); slack_var];
  expression lhs(N,K)
  expression rhs(N,K)
  for k = 1:K
    for n = 1:N
      lhs(n,k) = norm( diag( A((k-1)*N+n,:) ) * y );
      rhs(n,k) = (B((k-1)*N+n,:)*y - m_ik(n,k)) / conf(k);
    end
  end
  maximize( slack_var )
  subject to
    lhs <= rhs
    0 <= routing_vars <= 1
    sum( sum(routing_vars, 3), 2) <= 1
    sum( sum(routing_vars, 3), 1) <= 1
    slack_var >= slack_bound
    routing_vars(logical(repmat(eye(N), [1 1 K]))) == 0
cvx_end

status = ~isnan(slack_var); % check if a solution has been found