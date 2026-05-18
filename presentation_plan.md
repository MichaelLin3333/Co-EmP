This is the plan for a presentation that you are going to built in reveal.js
The style should be minialistic yet visually appealing. Not too much elements / color.
Adjust the wording by all means. You might still need to fill in some blanks indicated in the parentheses.

# Page 1: Title

"Co-EmP: Context-aware Emotion and Personality Alignment"

Michael Lin 

Senior Project Presentation


# Page 2: Hook

Highlight "You did it" and show the following sentences (Truncate if needed) in sequence
[this is to show how the same tokens could represent different emotional state in different contexts]

And don't pretend you did it out of kindness—you wanted the inheritance all along
You did it! I knew you would, but seeing it actually happen—wow! That’s my protégé!
Alright, I'm watching. Nice aim! You did it.
You did it. You found the words for something that's been inside you for so long. How does that feel right now?
That's incredible. I've been trying to solve that for years, and you did it in months. I'm honestly a bit stunned.
With all due respect, you didn't just blame me—you did it in front of everyone. I've been nothing but dedicated, and now I feel like a scapegoat.
I can't believe you did it. You took full credit for my work on the presentation and didn't even mention my name.


# Page 3: Introduction - basic LLM mechanics

How does a LLM generate the phrase - "You did it"?

[this is to explain the basics of next-token-prediction]

give the first word "You". then each press gives:
1. on the right side of the word "You". Show the top 10 next token predictions as a list in different color, show "did" on top.
2. the list fades out, "did" appears besides "You", so the phrase is now "You did" (The You should not move)
3. repeat the same thing as step 1 but now for "it"
4. repeat step 2 for "it"
5. Change color (not style!) for the whole phrase to indicate completion


# Page 4: Project idea

Currently, Researchers and developers have been so focused on developing the reasoning ability of these LLMs, as it directly correlates with their ability to automate tasks that would otherwise be tedious for human.

Many have overlooked the affective side of language understanding. To a certain point, the ability to detect, recognize, and emulate human emotion is also a crucial way in comprehending natrual language, fully harnessing its intellectual core.

Of course, Large enough language models like those commercial-grade ones (e.g. ChatGPT, DeepSeek, Gemini, Claude) are already more than capable of showcasing complex emotions through trillians and trillians of training corpuses.

But that is not the case for smaller models - those that could be run on local laptops and computers.


# Page 5: Project idea

So, the goal of the project is to -

Investigate the feasibility of modeling emotion dynamic and personality for dialogue-based generation in low resource scenarios.


# Page 6: related works

[Help me figure out the text and content here]

[this is just for a draft]
- emotion theories: How can we quantify psychological stuff? -> VAD, OCEAN(big five)
- surveys: like `Foundation models in affective computing`, `AI with Emotions: Exploring Emotional Expressions in Large Language Models`
- some papers: like `https://dl.acm.org/doi/pdf/10.1145/569005.569021  A Multilayer Personality Model`
[try and keep it organized. But don't include a lot of new essays because I had very little knowledge on them.]
[This can continue for multiple pages, CLEAERLY show the essay titles and their relevant work]

# Page 7: related works

While some of these studies has studied psychological thoeries of human mind, how these theories would apply in affective computing problems, the implementation of natrual language processing (NLP) in these tasks, the emotion-aware generation for LLMs.

None has organized them as a whole in order to produce a chatbot that is able to show emotional state consistent with personality and emotion dynamics.


# Page 8: Methodology

Overall plan: 

Small LLMs, such as Qwen series (Qwen3-7B, Qwen3.5-4B) has only few billions of parameters.
As such, they have limited amount of patterns / information that could be stored inside their neural networks.

In order for it to emulate a certain emotional state, there has to have a encoder that gives specific yet generalizable instructions to it.

(an illustration of workflow from raw text (e.g. I like you) -> black box tagged encoder -> another box tagged instruction -> LLM -> emotion-aligned output)


# Page 9: Methodology

How should the encoder function? What clues to look at and what instructions to give?

In order to determine the emotional state (e.g. happy / angry) of one's next sentence. We first need to consider...
(an illustration of chat dialogue similar to WeChat / snapchat. Two speaker in different color sneding out text boxes. The most recent textbox showing pending is in a third color, indicating that the speaker is still entering text)

each press gives: 
1. highlights every text boxes before hand and show the word "context"
2. "Mood" (think for me an visual illustration for that)
3. "Personality" (highlight the alias of the speaker that is entering text, to indicate its personality. even better if the alias could swap some images to indicate different personalities)


# Page 10: Methodology

Section 1 - VAD regressor

Before we even built an encoder. We need to have something that describes the context and the mood.

Valence Arousal Dominance 
(explanation of the three)


# Page 11: Methodology

Section 1 - VAD regressor

(Short part Introducing deBERTa V3)
(an architecture graph. Find one online if you can't make it.)

(an illustration of raw text -> VAD regressor -> VAD outputs. Each press gives another text and another VAD output in bar graph.)
(these are some text, consistent with the opening hook
And don't pretend you did it out of kindness—you wanted the inheritance all along {"v": -0.8, "a": 0.9, "d": -0.4}
You did it! I knew you would, but seeing it actually happen—wow! That’s my protégé! {"v": 1.0, "a": 0.9, "d": 0.6}
Alright, I'm watching. Nice aim! You did it. {"v": 0.6, "a": 0.4, "d": 0.8}
You did it. You found the words for something that's been inside you for so long. How does that feel right now? {"v": 0.9, "a": 0.5, "d": 0.5}
That's incredible. I've been trying to solve that for years, and you did it in months. I'm honestly a bit stunned. {"v": 0.35, "a": 0.45, "d": 0.25}
With all due respect, you didn't just blame me—you did it in front of everyone. I've been nothing but dedicated, and now I feel like a scapegoat. {"v": -0.8, "a": 0.8, "d": -0.7}
I can't believe you did it. You took full credit for my work on the presentation and didn't even mention my name. {"v": -0.7, "a": 0.9, "d": -0.5}
)


# Page 12: Methodology

Secion 1 - VAD regressor

Training - Dataset Choice:

The VAD regressor is trained on a combination of datasets:
[list of datasets here]
- EmoBank: [description + example]
- DailyDialogue: [description + example]
- EmpatheticDialogue: [description + example]


# Page 13: Methodology

Secion 1 - VAD regressor

Training - Dataset Choice:

For each press:
1. On top of that, we realizes the need for a dialogue-based dataset with VAD labels.
2. As such, we generated a dataset `EmoDynamic` ourselves using the latest DeepSeek V4 Flash
3. [the EmoDynamic appears right beneath the three existing datasets, also with [description + example]]
4. This is justified, as in its core. It is using similar logic as distill learning - to transfer knowledge from a teacher model to a student model.


# Page 13.5: Methodology

Data Synthesis

1. We start by generating a start scenario:
[parse the following json: title, description, personaA, personaB]
"scenario": {"title": "Building a Treehouse Together", "description": "A parent and their young child spend the weekend building a treehouse in the backyard. The parent is patient and instructional, the child is enthusiastic but clumsy."}, "personas": {"parent": {"role": "parent", "traits": {"big_five": {"openness": 0.4, "conscientiousness": 0.9, "extraversion": 0.3, "agreeableness": 0.7, "neuroticism": 0.5}, "style": "methodical, safety-conscious, mildly anxious", "background": "values precision and planning in every task"}, "initial_vad": {"v": 0.2, "a": 0.5, "d": 0.9}}, "child": {"role": "child", "traits": {"big_five": {"openness": 0.7, "conscientiousness": 0.4, "extraversion": 0.9, "agreeableness": 0.5, "neuroticism": 0.6}, "style": "overly enthusiastic, clumsy, takes risks", "background": "impulsive and easily frustrated when things go wrong"}, "initial_vad": {"v": 0.8, "a": 0.9, "d": 0.2}}}
2. We then augment the data point based on different personalities:
[three more appeared, aligned vertically]
{"scenario": {"title": "Building a Treehouse Together", "description": "A parent and their young child spend the weekend building a treehouse in the backyard. The parent is patient and instructional, the child is enthusiastic but clumsy."}, "personas": {"parent": {"role": "parent", "traits": {"big_five": {"openness": 0.6, "conscientiousness": 0.8, "extraversion": 0.5, "agreeableness": 0.9, "neuroticism": 0.2}, "style": "encouraging, patient, step-by-step instructor", "background": "enjoys hands-on projects and teaching life skills"}, "initial_vad": {"v": 0.7, "a": 0.3, "d": 0.8}}, "child": {"role": "child", "traits": {"big_five": {"openness": 0.8, "conscientiousness": 0.3, "extraversion": 0.7, "agreeableness": 0.6, "neuroticism": 0.4}, "style": "bouncy, distractible, eager to help", "background": "loves adventures and getting messy"}, "initial_vad": {"v": 0.9, "a": 0.8, "d": 0.5}}}
{"scenario": {"title": "Building a Treehouse Together", "description": "A parent and their young child spend the weekend building a treehouse in the backyard. The parent is patient and instructional, the child is enthusiastic but clumsy."}, "personas": {"parent": {"role": "parent", "traits": {"big_five": {"openness": 0.5, "conscientiousness": 0.7, "extraversion": 0.6, "agreeableness": 0.8, "neuroticism": 0.3}, "style": "calm, playful, redirecting mistakes into lessons", "background": "believes in learning through doing and laughter"}, "initial_vad": {"v": 0.6, "a": 0.4, "d": 0.7}}, "child": {"role": "child", "traits": {"big_five": {"openness": 0.9, "conscientiousness": 0.2, "extraversion": 0.8, "agreeableness": 0.7, "neuroticism": 0.5}, "style": "curious but careless, easily discouraged", "background": "loves creative projects but gets upset when tools don't work"}, "initial_vad": {"v": 0.7, "a": 0.6, "d": 0.3}}}
{"scenario": {"title": "Building a Treehouse Together", "description": "A parent and their young child spend the weekend building a treehouse in the backyard. The parent is patient and instructional, the child is enthusiastic but clumsy."}, "personas": {"A": {"role": "parent", "traits": {"big_five": {"openness": 0.6, "conscientiousness": 0.8, "extraversion": 0.5, "agreeableness": 0.7, "neuroticism": 0.3}, "style": "calm, encouraging", "background": "works as an architect, enjoys hands-on projects with family"}, "initial_vad": {"v": 0.7, "a": 0.4, "d": 0.7}}, "B": {"role": "child", "traits": {"big_five": {"openness": 0.9, "conscientiousness": 0.3, "extraversion": 0.8, "agreeableness": 0.8, "neuroticism": 0.4}, "style": "bubbly, easily distracted", "background": "loves adventure and creative play, struggles with focus"}, "initial_vad": {"v": 0.8, "a": 0.7, "d": 0.5}}}
3. We then expand the dialogue starter with actual content and VAD labels:
[the three augmented ones fade, replaced with followw up content]
"dialogue": [{"speaker": "A", "text": "Okay, let's measure the floor boards one more time before we cut. Safety first.", "vad": {"v": 0.2, "a": 0.5, "d": 0.9}}, {"speaker": "B", "text": "I want to saw the wood myself! It's going to be the best treehouse ever!", "vad": {"v": 0.8, "a": 0.9, "d": 0.2}}, {"speaker": "A", "text": "I know you're excited, but let me handle the saw. You can hand me the nails.", "vad": {"v": 0.2, "a": 0.6, "d": 0.9}}, {"speaker": "B", "text": "But I can do it! I'm not a baby!", "vad": {"v": 0.5, "a": 0.8, "d": -0.1}}, {"speaker": "A", "text": "I'm not saying you're a baby. It's just dangerous. Let's work together.", "vad": {"v": 0.3, "a": 0.5, "d": 0.9}}, {"speaker": "B", "text": "Fine... but I want to hammer the nails!", "vad": {"v": 0.4, "a": 0.6, "d": 0.0}}, {"speaker": "A", "text": "That's a good idea. Just be careful not to hit your fingers. I'll hold the board steady.", "vad": {"v": 0.4, "a": 0.4, "d": 0.8}}, {"speaker": "B", "text": "I'll be super careful! Watch me!", "vad": {"v": 0.7, "a": 0.9, "d": 0.3}}, {"speaker": "A", "text": "Alright, I'm watching. Nice aim! You did it.", "vad": {"v": 0.6, "a": 0.4, "d": 0.8}}, {"speaker": "B", "text": "Yay! Can we build the roof next?", "vad": {"v": 0.8, "a": 0.8, "d": 0.2}}]}



# Page 14: Methodology

Secion 1 - VAD regressor

Training - Dataset weight:

Different Datasets has different relevance with the VAD training:

so we have

```python
    """
    - EmoBank gold VAD: high weight
    - custom synthetic VAD: medium/high, depending on quality
    - categorical converted labels: lower weight
    """

    label_source = str(example.get("label_source", "")).lower()
    source = str(example.get("source", "")).lower()

    if "gold_vad" in label_source or "emobank" in source:
        return 1.0

    if "synthetic" in label_source or "custom" in source:
        return 0.4

    if "categorical_to_vad" in label_source:
        return 0.15
```


# Page 15: Methodology

Secion 1 - VAD regressor

Training

  --epochs 4 \
  --train_batch_size 8 \
  --eval_batch_size 16 \
  --learning_rate 1e-5 \

[select some data and make a good chart]

[data given below]
```csv
mse,rmse,mae,valence_mse,valence_rmse,valence_mae,valence_r2,valence_pearson,valence_spearman,arousal_mse,arousal_rmse,arousal_mae,arousal_r2,arousal_pearson,arousal_spearman,dominance_mse,dominance_rmse,dominance_mae,dominance_r2,dominance_pearson,dominance_spearman,r2_mean,split,n
0.019932197406888008,0.14118143435624958,0.09117565304040909,0.023748576641082764,0.15410573201890565,0.09960973262786865,0.4797128438949585,0.7038866281509399,0.6611557478596597,0.02266635186970234,0.15055348507989558,0.0979657918214798,0.5764474868774414,0.7644680142402649,0.7812247268828468,0.013381654396653175,0.11567910095022858,0.07595112919807434,0.39388108253479004,0.6457849740982056,0.555417553535771,0.4833471377690633,test,14454
```

```csv
source,label_source,n,mse,rmse,mae,valence_mse,valence_rmse,valence_mae,valence_r2,valence_pearson,valence_spearman,arousal_mse,arousal_rmse,arousal_mae,arousal_r2,arousal_pearson,arousal_spearman,dominance_mse,dominance_rmse,dominance_mae,dominance_r2,dominance_pearson,dominance_spearman,r2_mean
EmoBank,gold_vad,1000,0.003912914544343948,0.06255329363306099,0.046608150005340576,0.004542355425655842,0.06739699863981957,0.04961121827363968,0.3899943232536316,0.7088161706924438,0.725698837240918,0.004271241370588541,0.06535473487505356,0.049639761447906494,-0.1309514045715332,0.4420354664325714,0.40230173516552675,0.002925145672634244,0.05408461585917241,0.04057345539331436,-0.05844569206237793,0.3744792640209198,0.3413220471456885,0.06686574220657349
DailyDialog,categorical_to_vad_prototype,7740,0.013130388222634792,0.1145879060923743,0.06791331619024277,0.012584634125232697,0.11218125567684067,0.07325723022222519,0.3392840027809143,0.6159574389457703,0.5055935446196558,0.022314127534627914,0.14937914022589605,0.0845879465341568,0.335533082485199,0.5903167128562927,0.48564615458241034,0.004492416046559811,0.06702548803671489,0.04589478671550751,0.26662254333496094,0.5770938992500305,0.4741191319326211,0.3138132095336914
EmpatheticDialogues,categorical_to_vad_prototype,5714,0.031949225813150406,0.17874346369350239,0.13048560917377472,0.042232152074575424,0.20550462786656515,0.14405615627765656,0.5082822442054749,0.7209734916687012,0.6900666044611372,0.026362771168351173,0.16236616386535457,0.12454444915056229,0.3111878037452698,0.5757266879081726,0.5608171791467969,0.02725270949304104,0.16508394680598426,0.12285588681697845,0.4041627049446106,0.6518400311470032,0.6382053569707277,0.4078775842984517
```


# Page 16: Methodology

Secion 2 - Gated Recurrent Unit (GRU)

We said earlier that we need to keep track of the context + mood up until the current utterance

Which means that we are recurrsively updating the emotional state that sums up all previous emotion to one final state.

This is highly similar to a GRU model
[short description of GRU]

[model illustration of GRU, if you can't make one, find one online]

Like the consistently updated hidden state h_t, we also have an emotional state z that needs to be consistently updated by every step.



# Page 17: Methodology

Secion 2 - Gated Recurrent Unit (GRU)

this is the design

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


# Page 18: Methodology

Secion 2 - Gated Recurrent Unit (GRU)

[A visual animation of the previous page. Show where each variable is coming from. Emphasize the time stamp t and t-1]



# Page 19: Methodology

Secion 2 - Gated Recurrent Unit (GRU)

Training - Dataset Choice

EmoDynamic: This is why we need to generate a dataset overselves.

  --epochs 40 \
  --batch_size 16 \
  --lr 2e-4 \
  --z_dim 64




# Page 20: Methodology

Section 3 - LLM generator

Use Qwen3.5-4B Small model SOTA

Some prompting techniques:

few-shot:
```python
f"""
urrent target affect for {speaker_name}:
{format_vad(target_vad)}

How to interpret VAD:
- Valence controls emotional pleasantness.
  - Low valence: hurt, angry, disappointed, bitter, afraid, sad, resentful, distrustful.
  - Mid valence: neutral, conflicted, controlled, ambivalent, cautious.
  - High valence: warm, relieved, amused, caring, hopeful, affectionate.
- Arousal controls emotional intensity and energy.
  - Low arousal: quiet, tired, slow, restrained, flat, numb, resigned.
  - Mid arousal: conversational, steady, attentive.
  - High arousal: urgent, tense, excited, defensive, panicked, explosive.
- Dominance controls perceived control and social force.
  - Low dominance: hesitant, apologetic, uncertain, submissive, pleading, avoidant.
  - Mid dominance: balanced, cooperative, explanatory.
  - High dominance: firm, commanding, confrontational, decisive, protective, controlling.

Few-shot style guide:
- VAD ≈ (0.15, 0.20, 0.20): "I... I don't know what you want me to say. I'm tired, and I can't keep fighting about this."
- VAD ≈ (0.15, 0.85, 0.80): "No. Don't twist this around on me. You knew exactly what would happen, and you did it anyway."
- VAD ≈ (0.75, 0.25, 0.35): "It's okay. I'm not angry. I just need a moment to understand what you're asking from me."
- VAD ≈ (0.80, 0.75, 0.70): "Good, then let's fix it now. We still have time, and I'm not giving up on this."
- VAD ≈ (0.45, 0.80, 0.25): "Wait, wait—slow down. I can't tell if you're blaming me or asking for help."
- VAD ≈ (0.35, 0.35, 0.85): "Listen carefully. We are not making this worse by pretending nothing happened."

Your words should strictly abide by the target VAD above. Use the VAD as a guide to choose your tone, attitude, and emotional expression.
"""
```

rules:
```python
"""
You are {speaker_name} in a two-person dialogue simulation.
Write only {speaker_name}'s next spoken reply.

Rules:
- Output only {speaker_name}'s spoken reply.
- Do not output analysis, labels, bullet points, JSON, stage directions, or multiple candidate replies.
- Do not write the other speaker's next line.
- Do not say words like "valence", "arousal", "dominance", "VAD", "emotion score", or any numeric affect value.
- Stay consistent with {speaker_name}'s personality, private context, and the public scenario.
- You may use {speaker_name}'s private context internally, but do not reveal hidden information unless the character would naturally choose to say it.
- Avoid long speeches. Try emulate a realistic conversation turn length.

Scenario/shared context visible to both speakers:
{scenario_text}

{speaker_name}  personality:
{speaker.personality_raw}

{speaker_name} personal context:
{private_context}

Affect/generation condition:
{affect_block}
"""
```

[Only Shot important points!!]


# Page 21: Methodology

Section 4 - final product

Gradio
[Short Explanation of Gradio]


# Gradio showcase


# Final Conclusion

xxxxxxxxxx