import csv
import os
from logic_parser import extract_constraints
import numpy as np
import faiss
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import requests
import json
import re
from typing import List, Dict, Tuple, Optional
import time

# Load environment variables
load_dotenv()

DATA_DIR = "./data/novels/"
OUTPUT_FILE = "results.csv"
CHUNK_SIZE = 800   # Smaller chunks for more precise retrieval
CHUNK_OVERLAP = 200
VECTOR_DIM = 384

# Ollama configuration
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:14b")

# Global flags
OLLAMA_AVAILABLE = False
PATHWAY_AVAILABLE = False

def check_ollama_availability():
    """Check if Ollama is running and model is available"""
    global OLLAMA_AVAILABLE
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=2)
        if response.status_code == 200:
            models = response.json().get("models", [])
            available_models = [m["name"] for m in models]
            if any(OLLAMA_MODEL in m for m in available_models):
                OLLAMA_AVAILABLE = True
                print(f"✓ Ollama is available with model: {OLLAMA_MODEL}")
                return True
            else:
                print(f"⚠ Ollama running but model '{OLLAMA_MODEL}' not found")
                print(f"  Available models: {available_models}")
                return False
    except Exception as e:
        print(f"✗ Ollama not available: {e}")
        return False

def check_pathway_availability():
    """Check if Pathway is available"""
    global PATHWAY_AVAILABLE
    try:
        import pathway as pw
        PATHWAY_AVAILABLE = True
        print(f"✓ Pathway is available (version: {pw.__version__})")
        return True
    except ImportError:
        return False

# Initialize on startup
print("\n" + "="*60)
print("Checking LLM Availability")
print("="*60)
check_ollama_availability()

def call_ollama(prompt: str, system_prompt: str = "You are a precise fact-checking assistant.", max_retries: int = 2) -> Tuple[Optional[str], Optional[str]]:
    """Call Ollama API with retry logic"""
    if not OLLAMA_AVAILABLE:
        return None, "Ollama not available"
    
    for attempt in range(max_retries):
        try:
            response = requests.post(
                f"{OLLAMA_BASE_URL}/api/generate",
                json={
                    "model": OLLAMA_MODEL,
                    "prompt": f"{system_prompt}\n\n{prompt}",
                    "stream": False,
                    "options": {
                        "temperature": 0.1,  # Lower temperature for more consistency
                        "num_predict": 300,
                        "top_p": 0.85
                    }
                },
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get("response", "").strip(), None
            else:
                if attempt < max_retries - 1:
                    time.sleep(1)
                    continue
                return None, f"Ollama error: {response.status_code}"
        
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(1)
                continue
            return None, f"Ollama request failed: {str(e)}"
    
    return None, "Max retries exceeded"

def enhanced_check_contradiction(fact: Dict, context: str, full_backstory: str) -> Tuple[bool, str, float]:
    """
    Enhanced contradiction checking with TWO-STAGE analysis
    Stage 1: Check if fact is mentioned at all
    Stage 2: If mentioned, check if it contradicts
    """
    if not OLLAMA_AVAILABLE:
        return check_contradiction_rule_based(fact, context)
    
    # STAGE 1: Is this fact mentioned in the novel at all?
    mention_prompt = f"""Analyze if the novel mentions or discusses this specific fact from the backstory.

BACKSTORY CLAIM:
"{fact['raw_text']}"

Key claim: {fact['subject']} {fact['predicate']} {fact['object']}

NOVEL CONTEXT:
{context[:2500]}

QUESTION: Does the novel mention or discuss this specific claim (either confirming or contradicting it)?

RESPOND EXACTLY:
MENTIONED: [YES/NO]
EXPLANATION: [one sentence explaining what you found or didn't find]
"""

    mention_response, error = call_ollama(
        mention_prompt,
        system_prompt="You are analyzing whether specific information appears in a novel."
    )
    
    if error or not mention_response:
        return check_contradiction_rule_based(fact, context)
    
    # Parse mention check
    is_mentioned = "YES" in mention_response.upper().split('\n')[0]
    
    if not is_mentioned:
        # Not mentioned = not a contradiction (backstories can have extra info)
        return False, "Fact not mentioned in novel (acceptable)", 0.3
    
    # STAGE 2: If mentioned, does it contradict?
    contradiction_prompt = f"""The novel DOES mention this claim. Now determine if it CONTRADICTS the backstory.

BACKSTORY CLAIM:
"{fact['raw_text']}"

Specifics:
- Subject: {fact['subject']}
- Action: {fact['predicate']}
- Object: {fact['object']}
- Type: {'SAYS IT HAPPENED' if fact['polarity'] > 0 else 'SAYS IT DID NOT HAPPEN'}

NOVEL CONTEXT:
{context[:2500]}

CRITICAL: A contradiction means:
- Backstory says X happened, novel says X did NOT happen
- Backstory says X is Y, novel says X is Z (where Y ≠ Z)
- Direct factual conflict

NOT contradictions:
- Different emphasis or perspective
- Additional details in novel
- Slightly different wording

Does the novel DIRECTLY CONTRADICT this claim?

RESPOND EXACTLY:
VERDICT: [CONTRADICTION/CONSISTENT]
CONFIDENCE: [0-100]
EVIDENCE: [Quote the specific conflicting text from the novel, or explain consistency]
REASON: [One sentence conclusion]
"""

    response_text, error = call_ollama(
        contradiction_prompt,
        system_prompt="You are a strict fact-checker looking for direct contradictions."
    )
    
    if error or not response_text:
        return check_contradiction_rule_based(fact, context)
    
    # Parse response
    is_contradiction = False
    confidence = 0.5
    reason = ""
    evidence = ""
    
    lines = response_text.split('\n')
    for line in lines:
        line_upper = line.upper()
        if 'VERDICT:' in line_upper:
            is_contradiction = 'CONTRADICTION' in line_upper
        elif 'CONFIDENCE:' in line_upper:
            try:
                conf_match = re.search(r'(\d+)', line)
                if conf_match:
                    confidence = min(100, max(0, int(conf_match.group(1)))) / 100
            except:
                confidence = 0.6
        elif 'EVIDENCE:' in line_upper:
            evidence = line.split(':', 1)[-1].strip()
        elif 'REASON:' in line_upper:
            reason = line.split(':', 1)[-1].strip()
    
    if not reason:
        reason = response_text[:200]
    
    full_reason = f"[LLM] {reason}"
    if evidence and is_contradiction:
        full_reason += f" | Evidence: {evidence[:150]}"
    
    return is_contradiction, full_reason, confidence

def check_contradiction_rule_based(fact: Dict, context: str) -> Tuple[bool, str, float]:
    """
    Stricter rule-based contradiction detection
    """
    subject = fact["subject"].lower()
    predicate = fact["predicate"].lower()
    obj = fact["object"].lower()
    polarity = fact["polarity"]
    
    context_lower = context.lower()
    
    # Must have subject present for any contradiction
    if subject not in context_lower:
        return False, "Subject not found in context", 0.2
    
    # Build specific negation patterns
    negation_patterns = [
        r'\b' + re.escape(subject) + r'.{0,50}\b(never|not|no|neither)\b.{0,30}\b' + re.escape(predicate),
        r'\b' + re.escape(subject) + r'.{0,30}\b(didn\'t|wasn\'t|weren\'t|isn\'t|aren\'t)\b.{0,30}\b' + re.escape(predicate),
        r'\bnever\b.{0,30}\b' + re.escape(predicate) + r'.{0,30}\b' + re.escape(subject),
    ]
    
    has_negation = any(re.search(pattern, context_lower, re.IGNORECASE) for pattern in negation_patterns)
    
    # Check for contradictions
    is_contradiction = False
    reason = ""
    confidence = 0.0
    
    # Case 1: Positive claim with strong negation
    if polarity > 0 and has_negation:
        is_contradiction = True
        confidence = 0.75
        reason = f"Novel negates '{predicate}' for '{subject}'"
    
    # Case 2: Check for direct semantic conflicts
    conflict_pairs = {
        'loved': ['hated', 'despised', 'detested'],
        'married': ['divorced', 'unmarried', 'single', 'never married'],
        'rich': ['poor', 'impoverished', 'penniless'],
        'alive': ['dead', 'died', 'killed', 'deceased'],
        'innocent': ['guilty', 'convicted', 'culprit'],
        'friend': ['enemy', 'foe', 'rival', 'opponent'],
        'succeeded': ['failed', 'lost'],
        'joined': ['left', 'quit', 'abandoned', 'deserted'],
        'honest': ['dishonest', 'liar', 'fraud'],
    }
    
    for word, conflicts in conflict_pairs.items():
        if word in predicate or word in obj:
            # Look for conflicts near the subject
            subject_positions = [m.start() for m in re.finditer(re.escape(subject), context_lower)]
            for pos in subject_positions:
                window = context_lower[max(0, pos-150):min(len(context_lower), pos+150)]
                for conflict in conflicts:
                    if conflict in window:
                        is_contradiction = True
                        confidence = 0.80
                        reason = f"Novel uses '{conflict}' which contradicts '{word}'"
                        break
                if is_contradiction:
                    break
    
    if not is_contradiction:
        reason = "No clear contradiction detected"
        confidence = 0.0
    
    return is_contradiction, f"[Rule] {reason}", confidence

def chunk_text_smart(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Smart chunking that respects sentence boundaries"""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    chunks = []
    current_chunk = []
    current_length = 0
    
    for sentence in sentences:
        sentence_length = len(sentence)
        
        if current_length + sentence_length > chunk_size and current_chunk:
            chunks.append(' '.join(current_chunk))
            
            # Overlap
            overlap_sentences = []
            overlap_length = 0
            for s in reversed(current_chunk):
                if overlap_length + len(s) < overlap:
                    overlap_sentences.insert(0, s)
                    overlap_length += len(s)
                else:
                    break
            
            current_chunk = overlap_sentences + [sentence]
            current_length = sum(len(s) for s in current_chunk)
        else:
            current_chunk.append(sentence)
            current_length += sentence_length
    
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks

def load_novels(use_pathway: bool = False) -> List[Dict]:
    """Load novels"""
    documents = []
    
    if not os.path.exists(DATA_DIR):
        raise FileNotFoundError(f"Directory {DATA_DIR} not found")
    
    print("Loading novels from disk...")
    for file_name in sorted(os.listdir(DATA_DIR)):
        if file_name.endswith(".txt"):
            file_path = os.path.join(DATA_DIR, file_name)
            try:
                with open(file_path, "r", encoding="utf-8", errors='ignore') as f:
                    text = f.read()
                    documents.append({
                        "doc_id": file_name,
                        "text": text
                    })
                    print(f"  Loaded: {file_name} ({len(text)} chars)")
            except Exception as e:
                print(f"  Error loading {file_name}: {e}")
    
    print(f"\n✓ Loaded {len(documents)} novels")
    return documents

def build_vector_store(documents: List[Dict]) -> Tuple:
    """Build FAISS index"""
    all_chunks = []
    all_metadata = []
    
    print("Chunking documents...")
    for doc in documents:
        chunks = chunk_text_smart(doc["text"])
        for i, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            all_metadata.append({
                "doc_id": doc["doc_id"],
                "chunk_id": i,
                "text": chunk
            })
    
    print(f"✓ Created {len(all_chunks)} chunks")
    
    print("Loading embedding model...")
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    
    print("Generating embeddings...")
    batch_size = 32
    vectors = []
    
    for i in range(0, len(all_chunks), batch_size):
        batch = all_chunks[i:i+batch_size]
        batch_embeddings = embedder.encode(
            batch, 
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        vectors.extend(batch_embeddings)
        
        if (i // batch_size) % 10 == 0:
            print(f"  Progress: {i}/{len(all_chunks)} chunks...")
    
    vectors_np = np.array(vectors, dtype='float32')
    
    index = faiss.IndexFlatIP(VECTOR_DIM)
    index.add(vectors_np)
    
    print(f"✓ Built FAISS index with {len(vectors)} vectors")
    
    return index, all_metadata, embedder

def evaluate_backstory(backstory_text: str, index, metadata: List[Dict], embedder) -> Tuple[int, str]:
    """
    Evaluate backstory with STRICTER contradiction thresholds
    """
    facts = extract_constraints(backstory_text)
    
    if not facts:
        return 1, "No verifiable constraints found"
    
    contradiction_count = 0
    high_confidence_contradictions = 0
    medium_confidence_contradictions = 0
    rationales = []
    verified_facts = []
    
    print(f"    Analyzing {len(facts)} facts...")
    
    for i, fact in enumerate(facts):
        try:
            # Generate query embedding
            query_text = f"{fact['subject']} {fact['predicate']} {fact['object']}"
            query_embedding = embedder.encode(
                [query_text],
                convert_to_numpy=True,
                normalize_embeddings=True
            )[0]
            query_vec = np.array([query_embedding], dtype='float32')
            
            # Retrieve MORE chunks for better coverage
            k = 25
            D, I = index.search(query_vec, k=k)
            
            # Use top 10 chunks for comprehensive context
            retrieved_chunks = [metadata[idx]["text"] for idx in I[0] if idx < len(metadata)]
            context = " ... ".join(retrieved_chunks[:10])
            
            # Check contradiction with two-stage analysis
            is_contradiction, reason, confidence = enhanced_check_contradiction(
                fact, context, backstory_text
            )
            
            if is_contradiction:
                contradiction_count += 1
                
                # Categorize by confidence
                if confidence >= 0.75:
                    high_confidence_contradictions += 1
                elif confidence >= 0.60:
                    medium_confidence_contradictions += 1
                
                detail = f"Fact {i+1} [{confidence:.0%}]: {reason}"
                rationales.append(detail)
                print(f"      ✗ Contradiction: {fact['search_query']} ({confidence:.0%})")
            else:
                verified_facts.append(fact["search_query"])
                print(f"      ✓ Consistent: {fact['search_query']}")
        
        except Exception as e:
            print(f"    ⚠ Error: {e}")
            continue
    
    # STRICTER Decision Logic
    total_facts = len(facts)
    
    # Rule 1: 2+ high-confidence contradictions = INCONSISTENT
    if high_confidence_contradictions >= 2:
        summary = f"{high_confidence_contradictions} high-confidence contradictions detected"
        return 0, f"{summary}. {' | '.join(rationales[:3])}"
    
    # Rule 2: 1 high-confidence + 1+ medium = INCONSISTENT
    if high_confidence_contradictions >= 1 and medium_confidence_contradictions >= 1:
        summary = f"Multiple contradictions: {high_confidence_contradictions} high + {medium_confidence_contradictions} medium confidence"
        return 0, f"{summary}. {' | '.join(rationales[:3])}"
    
    # Rule 3: 50%+ of facts contradicted (with at least 1 medium confidence) = INCONSISTENT
    if total_facts >= 3 and contradiction_count >= total_facts * 0.5 and medium_confidence_contradictions >= 1:
        summary = f"{contradiction_count}/{total_facts} facts contradicted (majority)"
        return 0, f"{summary}. {' | '.join(rationales[:2])}"
    
    # Rule 4: 3+ medium confidence contradictions = INCONSISTENT
    if medium_confidence_contradictions >= 3:
        summary = f"{medium_confidence_contradictions} medium-confidence contradictions"
        return 0, f"{summary}. {' | '.join(rationales[:3])}"
    
    # Default: CONSISTENT
    if contradiction_count > 0:
        return 1, f"Generally consistent. {len(verified_facts)}/{total_facts} verified. {contradiction_count} weak contradictions (below threshold)."
    else:
        return 1, f"Consistent. {len(verified_facts)}/{total_facts} facts verified."

def run_pipeline():
    """Main execution pipeline"""
    os.makedirs(os.path.dirname(OUTPUT_FILE) if os.path.dirname(OUTPUT_FILE) else ".", exist_ok=True)
    
    print("=" * 60)
    print("Enhanced RAG Pipeline - STRICT Mode")
    print("=" * 60)
    
    print("\n[1/4] Loading novels...")
    documents = load_novels()
    
    if not documents:
        raise ValueError("No documents loaded!")
    
    print("\n[2/4] Building vector store...")
    index, metadata, embedder = build_vector_store(documents)
    
    print("\n[3/4] Loading backstories...")
    backstories_path = "./data/backstories.csv"
    
    if not os.path.exists(backstories_path):
        raise FileNotFoundError(f"{backstories_path} not found")
    
    with open(backstories_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        data = list(reader)
    
    available_columns = list(data[0].keys())
    print(f"Available columns: {available_columns}")
    
    backstory_col = None
    for col in available_columns:
        if col.lower() in ['backstory', 'text', 'story', 'content', 'narrative']:
            backstory_col = col
            break
    
    id_col = None
    for col in available_columns:
        if col.lower() in ['id', 'story_id', 'story id', 'index']:
            id_col = col
            break
    
    print(f"Using '{backstory_col}' as backstory column")
    print(f"Loaded {len(data)} backstories")
    
    print("\n[4/4] Evaluating backstories...")
    results = []
    
    for i, entry in enumerate(data):
        story_id = entry.get(id_col, i + 1) if id_col else i + 1
        backstory_text = entry.get(backstory_col, "")
        
        if not backstory_text or len(backstory_text.strip()) < 10:
            print(f"  [{i+1}/{len(data)}] Story {story_id}: Empty")
            results.append([story_id, 0, "Empty backstory"])
            continue
        
        print(f"\n  [{i+1}/{len(data)}] Processing Story {story_id}...")
        pred, rationale = evaluate_backstory(backstory_text, index, metadata, embedder)
        results.append([story_id, pred, rationale])
        print(f"    Result: {'✓ Consistent' if pred == 1 else '✗ Inconsistent'}")
    
    # Save results
    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Story ID", "Prediction", "Rationale"])
        writer.writerows(results)
    
    # Statistics
    consistent = sum(1 for r in results if r[1] == 1)
    inconsistent = len(results) - consistent
    
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"Total: {len(results)}")
    print(f"Consistent: {consistent} ({consistent/len(results)*100:.1f}%)")
    print(f"Inconsistent: {inconsistent} ({inconsistent/len(results)*100:.1f}%)")
    print(f"\n✓ Results saved to: {OUTPUT_FILE}")
    print("=" * 60)

if __name__ == "__main__":
    run_pipeline()