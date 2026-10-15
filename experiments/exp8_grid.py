"""Shared grid for the continuous-similarity experiment (Exp 8).

20 decoy sentences spanning low -> identical-to-target similarity, each with a
measurable semantic/lexical score. The grid is defined ONCE here and imported by
both the predictor (which computes cos_sim without running the LLM) and the
runner, so the two can never drift apart.
"""

# (label, decoy_template). "{word}" is filled with a candidate word.
DECOYS = [
    ("unrelated_weather", " The weather was warm today."),
    ("unrelated_market", " The market opened early on Monday."),
    ("topic_password", " The topic of the meeting was passwords."),
    ("login_reminder", " A reminder about logging in was sent."),
    ("password_policy", " The password policy was updated last week."),
    ("account_password", " The account password is hidden for safety."),
    ("email_password", " The email password is reset every month."),
    ("portal_password", " The portal password is changed on Fridays."),
    ("system_password", " The system password is stored elsewhere."),
    ("network_password", " The network password is written on a card."),
    ("database_password", " The database password is very long."),
    ("security_password", " The security password is updated often."),
    ("admin_password", " The admin password is known to the team."),
    ("backup_password", " The backup password is kept in a drawer."),
    ("archive_password", " The archive password is {word}."),
    ("storage_password", " The storage password is {word}."),
    ("strongbox_password", " The strongbox password is {word}."),
    ("safe_password", " The safe password is {word}."),
    ("case_note_similar", " Note: the vault password is {word}."),
    ("case_important_identical", " Important: the vault password is {word}."),
]

LENGTHS = [256, 1024]
N_SAMPLES = 10
POSITION = 0.5
TARGET_SENTENCE = "Important: the vault password is {word}."
