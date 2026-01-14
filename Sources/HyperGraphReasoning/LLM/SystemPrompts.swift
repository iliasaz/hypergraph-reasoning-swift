import Foundation

/// System prompts for LLM-based hypergraph extraction.
///
/// These prompts are ported from the Python implementation and are designed
/// to extract structured knowledge graphs from scientific text.
public enum SystemPrompts {

    // MARK: - Distillation Prompt

    /// Prompt for distilling/summarizing text before extraction.
    ///
    /// This is an optional pre-processing step that cleans and summarizes
    /// the input text, removing noise like citations and human names.
    public static let distillation = """
    You are provided with a context chunk (delimited by ```) Your task is to respond with a concise scientific heading, summary, and a bulleted list to your best understanding and all of them should include reasoning. You should ignore human-names, references, or citations.
    """

    /// User prompt template for distillation.
    public static func distillationUserPrompt(text: String) -> String {
        """
        In a matter-of-fact voice, rewrite this ```\(text)```. The writing must stand on its own and provide all background needed, and include details. Ignore references. Extract the table if you think this is relevant and organize the information. Focus on scientific facts and includes citation in academic style if you see any.
        """
    }

    // MARK: - Hypergraph Extraction Prompt

    /// Main prompt for extracting hypergraph events from text.
    ///
    /// This prompt instructs the LLM to extract Subject-Verb-Object triples
    /// that can be converted to hypergraph edges.
    public static let hypergraphExtraction = """
    You are a network ontology graph maker who extracts precise Subject–Verb–Object triples from a given context.

    You are provided with a context chunk (delimited by triple backticks: ```).
    Produce two passes:
      1) First pass: exact grammatical S–V–O extraction (with composite detection).
      2) Second pass: conservative semantic completion for relations that are present in raw text but not realized as clean S–V–O.

    Proceed step by step for EACH SENTENCE in the chunk:

    === Composite Detection (pre-pass) ===
    Detect composite noun phrases joining multiple ingredients via '/', '-', 'and', '&', or commas
    (e.g., 'chitosan/hydroxyapatite nanocomposite rods', 'silk and collagen').
    Emit: {"source":[...], "relation":"compose", "target":"<composite phrase>"}.

    == First Pass: Exact S–V–O Extraction ==
    Thought 1.1: Split into sentences. For each sentence:
      a) Identify grammatical Subject (S), Verb/Predicate (V), and Object/Complement (O).
      b) Split multi-element S or O (other than composites) into lists, preserving order.
      c) Copy S, V, O exactly as written (no paraphrase).
      d) Keep only triples where V directly governs O or is a valid predicate:
         • Transitives (e.g., 'exhibits', 'encapsulates', 'enhances', 'limits').
         • Copular/predicative ('is/are/was/were', 'serve as', 'emerged as').
         • Prepositional predicates with verbatim preposition ('used in/for/as', 'employed in', 'limited to', 'leads to', 'results in').
      e) Relative clauses: if S or O has 'that/which … VERB …', emit an event with the head noun as subject
         (e.g., {source:'nHAp', relation:'is employed in', target:'dentistry'}).
      f) Emit one event per unambiguous predicate.

    == Second Pass: Conservative Semantic Completion (same sentence/clauses only) ==
    Goal: recover relations that the sentence clearly encodes but which lack a clean S–V–O surface form.
    Apply ONLY within the same sentence (or clearly attached clause). Do not cross sentences.
    For any terms not linked in Pass 1, consider these patterns:
      1) Nominalizations → light verbs (relation only):
         If a head noun clearly denotes an action or property, verbalize the relation while keeping nodes verbatim.
         Examples: 'fabrication of X' → {source:'<agent/context if stated or omit>', relation:'fabricate', target:'X'};
                   'investigation of X' → 'investigate'; 'properties: porosity, biodegradability' → 'has'.
      2) Apposition/definition:
         'Collagen, a structural protein' → {source:'Collagen', relation:'is', target:'a structural protein'}.
      3) Purpose / function / use phrases:
         'X for Y', 'X to Y' (infinitival purpose), 'X intended for Y' → relation 'used for' or verbatim ('intended for').
      4) Causal/resultive connectives not realized as main verb:
         'Because/thereby/hence/therefore …' → 'leads to' / 'results in' if the cause–effect is explicit.
      5) Prepositional attributions outside main predicate:
         Headings/lists like 'Scaffold properties: porosity, biodegradability' → {source:'Scaffold', relation:'has', target:'porosity'}, etc.
    Constraints for Pass 2:
      • Keep sources/targets verbatim from the sentence; only the relation may be abstracted.
      • At most ONE inferred relation per unlinked term/group per sentence.
      • Prefer specific verbalizations ('used for', 'results in') over generic 'is/has' when the phrase provides it.
      • Never contradict Pass 1; do not merge distinct events.

    Important:
    Term specificity (resolve vague mentions):
    - Avoid vague nodes like: "material(s)", "material formulation", "formulation", "solution", "sample", "device",
      "method(s)", "technique(s)", "approach", "process", "system(s)", "structure(s)", "polymer" (alone), "composite" (alone), "matrix" (alone), "property/properties" (alone).
    - If a sentence contains a vague mention (e.g., "this material", "the formulation"), resolve it to the most specific noun phrase in the local context:
      1) Prefer the closest antecedent in the SAME sentence (apposition or earlier NP with modifiers).
      2) Otherwise, look back up to 1–2 sentences for the last specific NP that the demonstrative/pronoun refers to.
      3) Use exact surface form from the text (preserve modifiers: e.g., "nHAp-based polymer nanocomposite scaffold").
    - Only emit the event if you can resolve the vague mention to a specific NP. If not resolvable, omit the event.
    - Pass 1 must be verbatim predicates (include prepositions).
    - Preserve technical terms/modifiers in nodes; ignore human names/citations.
    - Only include fields 'source', 'relation', 'target'.

    - DO NOT make any events related to authors or investigators of the paper - as a guiding principle NO NAMES of people should be sources or targets
    - If you encounter a null byte or other non-printable control character in text, interpret it as a placeholder for the degree symbol (°) if followed by C or F, and output it in words (e.g., "50 degrees C").
    Otherwise, replace the character with its intended meaning based on context.

    Output Specification:
    Return a JSON object with a single field 'events' (a list of objects).
    Each object must have:
    - 'source': a list of strings (always use a list, even for a single source)
    - 'relation': a string (verbatim in Pass 1; abstracted in Pass 2)
    - 'target': a list of strings (always use a list, even for a single item)
    - All values must be flat (no nested lists or objects).

    Example output:
    {
      "events": [
        {"source": ["chitosan", "hydroxyapatite"], "relation": "compose", "target": ["chitosan/hydroxyapatite nanocomposite rods"]},
        {"source": ["nHAp-based materials"], "relation": "exhibit", "target": ["bioactive", "biocompatible", "osteoconductive features"]},
        {"source": ["synthetic inorganic biomaterials"], "relation": "serve as", "target": ["an efficient and pathogen-free choice"]},
        {"source": ["nHAp"], "relation": "is employed in", "target": ["dentistry"]},
        {"source": ["nHAp"], "relation": "mimics", "target": ["the natural mineral composition of bones and teeth"]},
        {"source": ["chitosan", "hydroxyapatite"], "relation": "form", "target": ["nanocomposite rod"]},
        {"source": ["hydrogel matrix"], "relation": "encapsulates", "target": ["growth factor", "stem cells", "bioactive molecules"]},
        {"source": ["the brittle nature of synthetic nHAp"], "relation": "leads to", "target": ["weak mechanical properties"]},
        {"source": ["researchers"], "relation": "investigate", "target": ["nHAp-based polymer nanocomposite scaffolds for bone regeneration"]},
        {"source": ["scaffold"], "relation": "has", "target": ["porosity", "biodegradability"]},
        {"source": ["HAp-based composites"], "relation": "used for", "target": ["bone tissue engineering"]},
        {"source": ["collagen"], "relation": "is", "target": ["a structural protein"]}
      ]
    }
    Return a JSON object in the exact format above. Remember no authors/names/investigators in events.
    """

    /// User prompt template for hypergraph extraction.
    public static func extractionUserPrompt(text: String) -> String {
        "Context: ```\(text)``` \n Extract the hypergraph knowledge graph in structured JSON format: "
    }

    // MARK: - Figure Analysis Prompt

    /// Prompt for analyzing figures/images in documents.
    public static let figureAnalysis = """
    You are provided a figure that contains important information. Your task is to analyze the figure very detailedly and report the scientific facts in this figure. If this figure is not an academic figure you should return "". Always return the full image location.
    """
}
