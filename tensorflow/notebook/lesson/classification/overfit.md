#overfiting problem
solve method:</br>

>1.**early termination**:stop train when validation accruacy become smaller.</br>
>2.**L2 regularization**:add constraint by hand.penalize the biger parameters by expression:</br>

$f'=f+\beta\frac{1}{2}||\omega||_2^2$

>3.**dropout**:make some active zero randomly.It make network more redundant,improve stability & reduce over-fiting.</br>
