WORDS_TO_SKIP = [
    "a",
    "axis",
    "abbreviate",
    "actually",
    "actual",
    "anatomy",
    "associated",
    "another",
    "anything",
    "adapt",
    "attachment",
    "about",
    "always",
    "able",
    "absorb",
    "absorbing",
    "all",
    "allow",
    "allowing",
    "accident",
    "atlas",
    "a little",
    "age",
    "any",
    "areas",
    "area",
    "as",
    "an",
    "are",
    "at",
    "area",
    "again",
    "and",
    "also",
    "back",
    "bad",
    "built",
    "basic",
    "bye",
    "big",
    "bit",
    "by",
    "but",
    "between",
    "bet",
    "better",
    "because",
    "be",
    "being",
    "become",
    "becomes",
    "becoming",
    "basically",
    "breakdown",
    "breaking",
    "broad",
    "broke",
    "broken",
    "branch",
    "branching",
    "both",
    "body",
    "bodies",
    "bother",
    "certain",
    "certainly",
    "comment",
    "comments",
    "create",
    "crawling",
    "cause",
    "class",
    "clients",
    "clinic",
    "close",
    "closing",
    "called",
    "can",
    "common",
    "come",
    "coming",
    "comes",
    "compare",
    "compact",
    "could",
    "count",
    "compared",
    "consider",
    "contract",
    "contracts",
    "character",
    "characteristics",
    "characteristic",
    "characteri",
    "contribution",
    "contributio",
    "contributi",
    "compromise",
    "considered",
    "cage",
    "diffirent",
    "diffirently",
    "diffirentl",
    "difficult",
    "difference",
    "direct",
    "directly",
    "did",
    "do",
    "down",
    "does",
    "done",
    "don\'t",
    "definite",
    "definitely",
    "depend",
    "depending",
    "disturb",
    "distinction",
    "distinctio",
    "disturbance",
    "either",
    "even",
    "ever",
    "end",
    "everybody",
    "everything",
    "earlier",
    "early",
    "early on",
    "each",
    "especially",
    "enjoy",
    "element",
    "extend",
    "extra",
    "example",
    "enter",
    "entering",
    "far",
    "function",
    "functional",
    "functions",
    "fits",
    "fitness",
    "fine",
    "first",
    "foreign",
    "for",
    "form",
    "forms",
    "focus",
    "front",
    "from",
    "feel",
    "free",
    "force",
    "forces",
    "guide",
    "give",
    "gives",
    "going",
    "go",
    "got",
    "gonna",
    "get",
    "give",
    "gave",
    "gets",
    "good",
    "great",
    "group",
    "green",
    "high",
    "highest",
    "helpful",
    "house",
    "housing",
    "hope",
    "hoping",
    "help",
    "helps",
    "helping",
    "here",
    "here\'s",
    "have",
    "half",
    "happen",
    "happens",
    "huge",
    "hugely",
    "having",
    "had",
    "has",
    "how",
    "hole",
    "holds",
    "hold",
    "health",
    "hard",
    "i",
    "is",
    "it",
    "its",
    "i\'ll",
    "isn\'t",
    "i\'ve",
    "it\'s",
    "in",
    "increase",
    "increasing",
    "is",
    "interesting",
    "interestin",
    "interest",
    "interested",
    "into",
    "intro",
    "introductory",
    "introducto",
    "introduction",
    "introducti",
    "introductio",
    "if",
    "instance",
    "impact",
    "injuries",
    "implement",
    "important",
    "imagine",
    "imagin",
    "just",
    "jump",
    "jumping",
    "know",
    "kind",
    "keep",
    "lead",
    "lecture",
    "learn",
    "little",
    "like",
    "likely",
    "low",
    "long",
    "lot",
    "lots",
    "let",
    "let\'s",
    "look",
    "looked",
    "looking",
    "later",
    "large",
    "line",
    "list",
    "mechanics",
    "make",
    "makes",
    "major",
    "maybe",
    "may be",
    "might",
    "more",
    "movement",
    "movements",
    "move",
    "moving",
    "movable",
    "my",
    "me",
    "minutes",
    "means",
    "mean",
    "most",
    "mostly",
    "much",
    "many",
    "material",
    "mainly",
    "main",
    "may",
    "mind",
    "mention",
    "mentioned",
    "named",
    "name",
    "names",
    "neat",
    "now",
    "no",
    "not",
    "neither",
    "nice",
    "new",
    "ones",
    "one\'s"
    "org",
    "organization",
    "organizati",
    "organizatio",
    "our",
    "off",
    "on",
    "of",
    "of the",
    "often",
    "out",
    "or",
    "over",
    "other",
    "overview",
    "obvious",
    "obviously",
    "ok",
    "okay",
    "opposite",
    "open",
    "opening",
    "old",
    "older",
    "phrase",
    "part",
    "patients",
    "patient",
    "people",
    "people\'s",
    "please",
    "play",
    "planes",
    "plane",
    "primary"
    "practice",
    "practition",
    "process",
    "processes",
    "probably",
    "protect",
    "project",
    "projects",
    "put",
    "puts",
    "point",
    "points",
    "prone",
    "picture",
    "place",
    "placing",
    "pretty",
    "period",
    "periods",
    "person",
    "quite",
    "quick",
    "questions",
    "question",
    "region",
    "regions",
    "remember",
    "related",
    "referral",
    "refer",
    "referred",
    "related",
    "really",
    "real",
    "run",
    "running",
    "right",
    "service",
    "services",
    "side",
    "short",
    "shorter",
    "stack",
    "stop",
    "stacked",
    "sorry",
    "source",
    "say",
    "said",
    "same",
    "see",
    "sometimes",
    "somewhere",
    "someone",
    "somebody",
    "somebody\'s",
    "so",
    "soon",
    "sooner",
    "some",
    "something",
    "sort",
    "start",
    "specific",
    "study",
    "studying",
    "show",
    "showing",
    "saw",
    "secondary",
    "second",
    "see",
    "stands",
    "site",
    "structure",
    "structural",
    "state",
    "status",
    "section",
    "small",
    "smaller",
    "size",
    "since",
    "slide",
    "super",
    "surgeons",
    "transmit",
    "trim",
    "take",
    "talk",
    "talking",
    "talked",
    "than",
    "top",
    "to",
    "tons",
    "ton",
    "together",
    "there",
    "them",
    "then",
    "the",
    "their",
    "these",
    "those",
    "thought",
    "though",
    "that",
    "that\'s",
    "there\'s",
    "they\'ll",
    "they\'re",
    "think",
    "thinking",
    "this",
    "that",
    "today",
    "too",
    "term",
    "terms",
    "time",
    "thing",
    "things",
    "they",
    "through",
    "throughout",
    "type",
    "technicall",
    "technical",
    "technique",
    "techniques",
    "technically",
    "tight",
    "tighter",
    "us",
    "up",
    "use",
    "uh",
    "um",
    "unit",
    "under",
    "understand",
    "understanding",
    "underneath",
    "usually",
    "vertebrae",
    "versus",
    "variable",
    "variability",
    "work",
    "word",
    "were",
    "wear",
    "welcome",
    "was",
    "with",
    "without",
    "way",
    "we",
    "who",    
    "whole",    
    "where",
    "what",
    "we\'ll",
    "we\'ve",
    "we\'re",
    "will",
    "wanted",
    "when",
    "want",
    "would",
    "which",
    "very",
    "you",
    "you\'re",
    "you\'ve",
    "your",
    "you know",
    "yet",
    "yeah",
    "yes",
    "yesterday",
    "youtube",
    # Add more words here
]
