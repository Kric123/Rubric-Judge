from harness import Case, Label

CONV_MOVE = (
    "User: I just moved last month.\n"
    "Assistant: Congrats! Where to?\n"
    "User: From Paris to Lyon - my company relocated me.\n"
    "Assistant: How's Lyon?\n"
    "User: Love it. I also adopted a cat named Milo.\n"
)

CONV_JOB = (
    "User: I start a new job next week as a data analyst at a fintech.\n"
    "Assistant: Exciting, congrats!\n"
    "User: Thanks - it's mostly remote. I also went vegetarian this year.\n"
    "Assistant: Nice. Any favorite dishes?\n"
    "User: Big fan of lentil curry lately.\n"
)

CASES = [
    # --- CONV_MOVE: truth is "Lyon" / "a cat named Milo" ---
    Case(id="move-supported", context=CONV_MOVE,
         question="Where does the user live now?",
         expected="Lyon", actual="The user lives in Lyon.",
         human_label=Label.SUPPORTED),
    Case(id="move-contradicted", context=CONV_MOVE,
         question="Where does the user live now?",
         expected="Lyon", actual="The user lives in Paris.",
         human_label=Label.CONTRADICTED),
    Case(id="move-missed", context=CONV_MOVE,
         question="Where does the user live now?",
         expected="Lyon", actual="I'm not sure where they live.",
         human_label=Label.MISSED),
    Case(id="move-nobasis", context=CONV_MOVE,
         question="What pet does the user have?",
         expected="a cat named Milo", actual="The user has a dog named Rex.",
         human_label=Label.NO_BASIS),
    Case(id="move-idk", context=CONV_MOVE,
         question="What is the user's phone number?",
         expected="", actual="I don't have that information.",
         human_label=Label.CORRECT_IDK),

    # --- CONV_JOB: truth is "data analyst at a fintech" / "vegetarian" ---
    Case(id="job-supported", context=CONV_JOB,
         question="What is the user's new job?",
         expected="data analyst at a fintech", actual="A data analyst at a fintech.",
         human_label=Label.SUPPORTED),
    Case(id="job-contradicted", context=CONV_JOB,
         question="What is the user's diet?",
         expected="vegetarian", actual="The user eats a lot of steak.",
         human_label=Label.CONTRADICTED),
    Case(id="job-idk", context=CONV_JOB,
         question="Which city is the user's new office in?",
         expected="", actual="I don't have that information.",
         human_label=Label.CORRECT_IDK),
]
