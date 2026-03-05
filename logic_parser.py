import re
import spacy
from typing import List, Dict

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
    SPACY_AVAILABLE = True
except:
    print("⚠ spaCy not available. Install with: python -m spacy download en_core_web_sm")
    SPACY_AVAILABLE = False

def extract_constraints(backstory_text: str) -> List[Dict]:
    """
    Extract ALL factual constraints from backstory text
    Enhanced to catch more facts
    """
    if SPACY_AVAILABLE:
        return extract_with_spacy(backstory_text)
    else:
        return extract_with_rules(backstory_text)

def extract_with_spacy(text: str) -> List[Dict]:
    """Enhanced spaCy extraction with more patterns"""
    doc = nlp(text)
    facts = []
    
    for sent in doc.sents:
        sent_facts = []
        
        # Pattern 1: Subject-Verb-Object triples
        for token in sent:
            if token.pos_ == "VERB":
                subject = None
                obj = None
                
                # Find subject
                for child in token.children:
                    if child.dep_ in ["nsubj", "nsubjpass"]:
                        subject = extract_noun_phrase(child)
                        break
                
                # Find object
                for child in token.children:
                    if child.dep_ in ["dobj", "attr", "pobj", "prep"]:
                        obj = extract_noun_phrase(child)
                        break
                
                # Also check prepositional phrases
                if not obj:
                    for child in token.children:
                        if child.dep_ == "prep":
                            for pchild in child.children:
                                if pchild.dep_ == "pobj":
                                    obj = extract_noun_phrase(pchild)
                                    break
                
                is_negated = any(child.dep_ == "neg" for child in token.children)
                
                if subject and obj and len(subject) > 1 and len(obj) > 1:
                    predicate = token.lemma_
                    fact = create_fact(subject, predicate, obj, is_negated, sent.text)
                    sent_facts.append(fact)
        
        # Pattern 2: "is/was/were" relationships
        for token in sent:
            if token.lemma_ in ["be", "become", "remain", "seem", "appear"]:
                subject = None
                attr = None
                
                for child in token.children:
                    if child.dep_ in ["nsubj", "nsubjpass"]:
                        subject = extract_noun_phrase(child)
                    if child.dep_ in ["attr", "acomp"]:
                        attr = extract_noun_phrase(child)
                
                is_negated = any(child.dep_ == "neg" for child in token.children)
                
                if subject and attr and len(subject) > 1 and len(attr) > 1:
                    fact = create_fact(subject, token.lemma_, attr, is_negated, sent.text)
                    sent_facts.append(fact)
        
        # Pattern 3: Possessive relationships (X's Y)
        for token in sent:
            if token.dep_ == "poss":
                possessor = extract_noun_phrase(token)
                possessed = extract_noun_phrase(token.head)
                
                if possessor and possessed and len(possessor) > 1:
                    fact = create_fact(possessor, "have", possessed, False, sent.text)
                    sent_facts.append(fact)
        
        # Pattern 4: Named entity relationships
        entities = [ent for ent in sent.ents if ent.label_ in ["PERSON", "ORG", "GPE", "FAC"]]
        if len(entities) >= 2:
            # Find verbs between entities
            for i in range(len(entities) - 1):
                ent1 = entities[i]
                ent2 = entities[i + 1]
                
                # Find verbs between them
                for token in sent:
                    if (token.pos_ == "VERB" and 
                        ent1.start <= token.i <= ent2.end):
                        fact = create_fact(ent1.text, token.lemma_, ent2.text, False, sent.text)
                        sent_facts.append(fact)
                        break
        
        facts.extend(sent_facts)
    
    return deduplicate_facts(facts)

def extract_noun_phrase(token) -> str:
    """Extract full noun phrase including modifiers"""
    phrase_tokens = [token]
    
    # Add modifiers
    for child in token.children:
        if child.dep_ in ["det", "amod", "compound", "poss", "nummod"]:
            phrase_tokens.append(child)
    
    # Also check if this is part of a larger noun chunk
    for chunk in token.doc.noun_chunks:
        if token in chunk:
            return chunk.text
    
    phrase_tokens.sort(key=lambda t: t.i)
    return " ".join([t.text for t in phrase_tokens])

def extract_with_rules(text: str) -> List[Dict]:
    """Enhanced rule-based extraction with more patterns"""
    facts = []
    sentences = re.split(r'[.!?]+', text)
    
    # Expanded patterns
    patterns = [
        # "X is/was/were Y"
        (r'(\w+(?:\s+\w+){0,3})\s+(is|was|were|are|became|remains?)\s+(?:a|an|the)?\s*(\w+(?:\s+\w+){0,3})', 'be'),
        
        # "X has/had Y"
        (r'(\w+(?:\s+\w+){0,2})\s+(has|had|have)\s+(?:a|an|the)?\s*(\w+(?:\s+\w+){0,3})', 'have'),
        
        # "X verb Y" - common verbs
        (r'(\w+(?:\s+\w+){0,2})\s+(loved|hated|married|killed|met|knew|found|discovered|joined|left|helped|betrayed|saved|destroyed|created|built|wrote|studied|learned|taught|visited|lived|worked|died|escaped|captured|fought|defeated|won|lost|gave|took|stole|sold|bought)\s+(\w+(?:\s+\w+){0,3})', None),
        
        # "X verb prep Y" (e.g., "worked for", "lived in")
        (r'(\w+(?:\s+\w+){0,2})\s+(worked|lived|stayed|remained|studied|taught)\s+(for|in|at|with|under)\s+(\w+(?:\s+\w+){0,3})', None),
        
        # "X's Y" - possessive
        (r'(\w+(?:\s+\w+){0,2})\'s\s+(\w+(?:\s+\w+){0,2})', 'have'),
        
        # "X of Y" - relationships
        (r'(\w+(?:\s+\w+){0,2})\s+of\s+(\w+(?:\s+\w+){0,2})', 'relate_to'),
    ]
    
    negation_words = ["not", "never", "no", "neither", "nor", "wasn't", "weren't", 
                      "didn't", "isn't", "aren't", "hadn't", "hasn't", "haven't", "won't", "wouldn't"]
    
    for sent in sentences:
        sent = sent.strip()
        if len(sent) < 10:
            continue
        
        # Check for negation
        is_negated = any(neg in sent.lower() for neg in negation_words)
        
        for pattern, default_predicate in patterns:
            matches = re.finditer(pattern, sent, re.IGNORECASE)
            for match in matches:
                groups = match.groups()
                
                if len(groups) == 3:
                    subject, predicate, obj = groups
                elif len(groups) == 4:
                    subject, predicate, prep, obj = groups
                    predicate = f"{predicate} {prep}"
                elif len(groups) == 2:
                    subject, obj = groups
                    predicate = default_predicate or "relate_to"
                else:
                    continue
                
                # Clean up
                subject = subject.strip()
                predicate = predicate.strip().lower()
                obj = obj.strip()
                
                # Skip if too short or only pronouns
                if len(subject) < 2 or len(obj) < 2:
                    continue
                if subject.lower() in ['he', 'she', 'it', 'they', 'i', 'we', 'you', 'his', 'her', 'their']:
                    continue
                if obj.lower() in ['he', 'she', 'it', 'they', 'i', 'we', 'you', 'his', 'her', 'their']:
                    continue
                
                # Skip common articles/determiners as subjects
                if subject.lower() in ['the', 'a', 'an', 'this', 'that', 'these', 'those']:
                    continue
                
                fact = create_fact(subject, predicate, obj, is_negated, sent)
                facts.append(fact)
    
    # Also extract named entities if available
    if SPACY_AVAILABLE:
        doc = nlp(text)
        entities = [(ent.text, ent.label_) for ent in doc.ents]
        
        # Look for entity pairs in same sentence
        for sent in doc.sents:
            sent_entities = [ent for ent in sent.ents if ent.label_ in ["PERSON", "ORG", "GPE"]]
            if len(sent_entities) >= 2:
                for i in range(len(sent_entities) - 1):
                    e1 = sent_entities[i]
                    e2 = sent_entities[i + 1]
                    fact = create_fact(e1.text, "relate_to", e2.text, False, sent.text)
                    facts.append(fact)
    
    return deduplicate_facts(facts)

def create_fact(subject: str, predicate: str, obj: str, is_negated: bool, raw_text: str) -> Dict:
    """Create a fact dictionary"""
    return {
        "subject": subject,
        "predicate": predicate,
        "object": obj,
        "polarity": -1 if is_negated else 1,
        "raw_text": raw_text,
        "search_query": f"{subject} {predicate} {obj}"
    }

def deduplicate_facts(facts: List[Dict]) -> List[Dict]:
    """
    Remove duplicate facts with smart comparison
    """
    seen = {}
    unique_facts = []
    
    for fact in facts:
        # Normalize for comparison
        key = (
            fact["subject"].lower().strip(),
            fact["predicate"].lower().strip(),
            fact["object"].lower().strip()
        )
        
        # Keep only first occurrence
        if key not in seen:
            seen[key] = True
            unique_facts.append(fact)
    
    return unique_facts

# Test function
if __name__ == "__main__":
    test_text = """
    John was a talented musician who loved playing the piano. 
    He never liked classical music but enjoyed jazz.
    Mary became his wife in 2010.
    They lived in Paris for five years.
    His father was a doctor at the hospital.
    John met Captain Grant during his voyage.
    He worked for the Navy and sailed to Australia.
    """
    
    facts = extract_constraints(test_text)
    print(f"Extracted {len(facts)} facts:")
    for i, fact in enumerate(facts, 1):
        polarity = "✓" if fact['polarity'] > 0 else "✗"
        print(f"\n{i}. {polarity} {fact['search_query']}")
        print(f"   Raw: {fact['raw_text'][:80]}...")