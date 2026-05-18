$z_A(t) = GRU(x(t),\:z_A(t-1))$

$x(t) = [v_A(t-1),\;v_B(t),\;v_B(t)-v_A(t-1),\;p_A]$

$v_A(t) = g(x(t))$

where \
$z_A(t)$ is the emotional state embedding at time stamp $t$ \
$v_A(t-1)$ is the predicted VAD metric for the previous utterance \
$v_B(t)$ is the VAD metric produced by the VAD regressor \
$p_A$ is the personality metric in Big Five (OCEAN) \
$g$ is a Multi-layer Perceptron (MLP)
$v_A(t)$ is the predicted VAD metric for the target utterance \



$z_A(t) = GRU(x(t),\:z_A(t-1))$

$x(t) = [v_A(t-1),\;v_B(t),\;v_B(t)-v_A(t-1),\;p_A,\;a_{B \to A}(t)]$

$a_{B \to A}(t) = f(u_B(t))$

where \
$a_{B \to A}(t)$ is the appraisal vector \
$f$ is a deBERTa V3 decoder \
$u_B(t)$ is the raw text for the current utterance speaker B said, which speaker A is going to repond to.


Another feature:
Now set the entire page into 4 modes:
1) pure baseline mode
2) emotion supported mode (with the projects' stuff)
3) arena comparison (where we are at now)
4) two AI mode (AI plays both role of speakerA and speakerB. each taking GRU analysis on another, each with their own context, etc. Be special careful not to confuse between different speakers' variables)

Now, I need another two context boxes, where each speaker gets their own individual context to provide information to the LLM. The context for speaker A is for speaker A only and would not be shown for the other speaker. Same applies to B (even though user now is playing A)