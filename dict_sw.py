# dict_LIGHTWEIGHT_NO_HEADINGS.py — Schwartz-Hearst with heading removal

import os
import pandas as pd
import fitz
import re
import time
import stat
from collections import defaultdict

def safe_delete(file_path, retries=5, delay=1):
    for attempt in range(retries):
        try:
            if os.path.exists(file_path):
                os.chmod(file_path, stat.S_IWRITE)
                os.remove(file_path)
                print(f"✓ Deleted: {os.path.basename(file_path)}")
            break
        except PermissionError:
            print(f"⚠ Retrying...")
            time.sleep(delay)
        except Exception as e:
            print(f"✗ Error: {e}")
            break


class SchwartzHearstExtractor:
    """
    Lightweight implementation of Schwartz-Hearst algorithm
    with heading detection and removal
    """
    
    def __init__(self, input_folder, output_folder="output_lightweight"):
        self.input_folder = input_folder
        self.output_folder = output_folder
        os.makedirs(self.output_folder, exist_ok=True)
        
        # Storage
        self.acronym_mappings = {}
        self.acronym_frequencies = defaultdict(int)
        self.standalone_acronyms = defaultdict(int)
        
        # Output files
        self.mapping_csv = os.path.join(self.output_folder, "acronym_mappings.csv")
        self.standalone_csv = os.path.join(self.output_folder, "standalone_acronyms.csv")
        
        # Common stop words to skip
        self.stop_words = {'the', 'a', 'an', 'and', 'or', 'of', 'for', 'in', 'on', 'at', 'to', 'by', 'with'}
        
        # Heading keywords
        self.heading_keywords = [
            'chapter', 'section', 'part', 'article', 'annexure', 
            'appendix', 'volume', 'abstract', 'introduction', 
            'conclusion', 'summary', 'references', 'index', 
            'contents', 'session', 'appendices', 'preface',
            'foreword', 'acknowledgment', 'bibliography'
        ]
    
    def is_heading(self, line):
        """
        Detect if a line is a heading based on multiple criteria
        """
        if not line or len(line.strip()) == 0:
            return False
        
        line = line.strip()
        words = line.split()
        
        if len(words) == 0:
            return False
        
        # 1. All uppercase lines with 2-12 words (typical headings)
        if line.isupper() and 2 <= len(words) <= 12:
            return True
        
        # 2. Lines ending with colon (section headers)
        if line.endswith(':'):
            return True
        
        # 3. Numbered sections (1., 1.1, I., etc.)
        if re.match(r'^(\d+\.|\d+\.\d+|[IVXLCDM]+\.)\s+', line):
            return True
        
        # 4. Title case with most words capitalized (70%+)
        if 2 <= len(words) <= 8:
            cap_count = sum(1 for w in words if w and len(w) > 0 and w[0].isupper())
            if cap_count / len(words) >= 0.7:
                return True
        
        # 5. Contains heading keywords
        line_lower = line.lower()
        for keyword in self.heading_keywords:
            if line_lower.startswith(keyword):
                return True
        
        return False
    
    def remove_headings_from_text(self, text):
        """
        Remove all heading lines from text before processing
        """
        lines = text.split('\n')
        clean_lines = []
        
        removed_count = 0
        for line in lines:
            if not self.is_heading(line):
                clean_lines.append(line)
            else:
                removed_count += 1
        
        print(f"  Removed {removed_count} heading lines")
        return '\n'.join(clean_lines)
    
    def extract_text_from_pdf(self, pdf_path):
        """Extract text from PDF"""
        try:
            doc = fitz.open(pdf_path)
            text = ""
            for page in doc:
                text += page.get_text("text") + "\n"
            doc.close()
            return text
        except Exception as e:
            print(f"Error extracting text: {e}")
            return ""
    
    def is_valid_acronym(self, text):
        """Check if text is a valid acronym"""
        if not text:
            return False
        if not text.isupper():
            return False
        if len(text) < 2 or len(text) > 10:
            return False
        # Must be alphabetic
        if not text.isalpha():
            return False
        # Exclude single repeated letters
        if len(set(text)) == 1 and len(text) <= 2:
            return False
        # Exclude common false positives
        false_positives = ['NO', 'YES', 'OK', 'AM', 'PM', 'AD', 'BC']
        if text in false_positives:
            return False
        return True
    
    def is_likely_heading_word(self, word):
        """
        Check if a single word is likely part of a heading
        (all caps multi-word phrases)
        """
        # If word is short (2-3 letters) and all caps, might be heading
        # But we need context, so this is used in combination with other checks
        return False  # Handled by line-level heading detection
    
    def extract_min_max(self, short_form, long_form):
        """
        Schwartz-Hearst algorithm core:
        Determine the minimum and maximum boundaries of the long form
        """
        sf_len = len(short_form)
        lf_len = len(long_form)
        
        # Long form must be at least as long as short form
        if lf_len < sf_len:
            return None, None
        
        # Long form cannot be more than min(len(SF) + 5, len(SF) * 2) words
        lf_words = long_form.split()
        max_words = min(sf_len + 5, sf_len * 2)
        
        if len(lf_words) > max_words:
            # Take only the last max_words
            long_form = ' '.join(lf_words[-max_words:])
        
        return 0, len(long_form)
    
    def find_best_long_form(self, short_form, long_form_candidate):
        """
        Implements the character matching part of Schwartz-Hearst
        """
        # Clean up
        short_form = short_form.strip()
        long_form = long_form_candidate.strip()
        
        # Get boundaries
        min_idx, max_idx = self.extract_min_max(short_form, long_form)
        if min_idx is None:
            return None
        
        # Try to match short form letters to long form
        sf_index = len(short_form) - 1
        lf_index = len(long_form) - 1
        
        while sf_index >= 0:
            curr_char = short_form[sf_index].lower()
            
            # Find this character in long form (going backwards)
            found = False
            while lf_index >= 0:
                if long_form[lf_index].lower() == curr_char:
                    found = True
                    lf_index -= 1
                    break
                lf_index -= 1
            
            if not found:
                return None
            
            sf_index -= 1
        
        # Extract the matched portion
        # Find the start of the long form (first capital letter or word boundary)
        start_pos = lf_index + 1
        
        # Look for word boundary
        while start_pos > 0 and long_form[start_pos - 1] not in ' \t\n':
            start_pos -= 1
        
        matched_long_form = long_form[start_pos:].strip()
        
        return matched_long_form if len(matched_long_form) > 0 else None
    
    def extract_candidates_from_parentheses(self, text):
        """
        Find all patterns of: text (ACRONYM)
        """
        candidates = []
        
        # Pattern: (UPPERCASE) where UPPERCASE is 2-10 letters
        pattern = r'([^(]{0,200})\(([A-Z]{2,10})\)'
        
        for match in re.finditer(pattern, text):
            long_form_candidate = match.group(1).strip()
            short_form = match.group(2).strip()
            
            if not self.is_valid_acronym(short_form):
                continue
            
            # Clean long form: remove text before last sentence boundary
            sentence_markers = ['.', '!', '?', ';', ':']
            for marker in sentence_markers:
                last_pos = long_form_candidate.rfind(marker)
                if last_pos != -1:
                    long_form_candidate = long_form_candidate[last_pos + 1:].strip()
                    break
            
            if long_form_candidate:
                candidates.append((short_form, long_form_candidate))
        
        return candidates
    
    def clean_long_form(self, long_form):
        """Clean extracted long form"""
        if not long_form:
            return ""
        
        # Split into words
        words = long_form.split()
        
        # Remove leading stop words and non-capitalized words
        while words and (words[0].lower() in self.stop_words or not words[0][0].isupper()):
            words.pop(0)
        
        # Remove trailing punctuation
        if words:
            words[-1] = words[-1].rstrip('.,;:!?')
        
        result = ' '.join(words)
        
        # Must start with capital
        if result and result[0].islower():
            result = result[0].upper() + result[1:]
        
        return result
    
    def validate_long_form(self, short_form, long_form):
        """Validate that long form is reasonable"""
        if not long_form or len(long_form) < 3:
            return False
        
        words = long_form.split()
        
        # Must have at least 2 words
        if len(words) < 2:
            return False
        
        # Check if it's likely a heading (all caps multi-word)
        # If full form is all uppercase with 3+ words, likely a heading
        if long_form.isupper() and len(words) >= 3:
            return False
        
        # Must not be a sentence (no verbs like "has established", "is the")
        long_form_lower = long_form.lower()
        verb_indicators = [' has ', ' have ', ' is ', ' are ', ' was ', ' were ', ' established ', ' created ']
        for indicator in verb_indicators:
            if indicator in long_form_lower:
                return False
        
        # At least 40% of words should be capitalized
        cap_count = sum(1 for w in words if w and w[0].isupper())
        if cap_count / len(words) < 0.4:
            return False
        
        # Check acronym-to-longform initial matching (loose check)
        initials = ''.join([w[0].upper() for w in words if w and w[0].isalpha()])
        
        if len(initials) >= 2:
            # At least 1/3 of acronym letters should match
            matches = 0
            j = 0
            for char in short_form:
                while j < len(initials):
                    if initials[j] == char:
                        matches += 1
                        j += 1
                        break
                    j += 1
            
            match_ratio = matches / len(short_form) if len(short_form) > 0 else 0
            if match_ratio < 0.3:
                return False
        
        return True
    
    def extract_acronyms(self, text):
        """Main extraction method using Schwartz-Hearst algorithm"""
        print("Extracting acronyms using Schwartz-Hearst algorithm...")
        
        # IMPORTANT: Remove headings first
        text = self.remove_headings_from_text(text)
        
        # Find candidates
        candidates = self.extract_candidates_from_parentheses(text)
        
        print(f"  Found {len(candidates)} candidates")
        
        # Process each candidate
        for short_form, long_form_candidate in candidates:
            # Apply Schwartz-Hearst matching
            long_form = self.find_best_long_form(short_form, long_form_candidate)
            
            if not long_form:
                continue
            
            # Clean the long form
            long_form = self.clean_long_form(long_form)
            
            # Validate
            if not self.validate_long_form(short_form, long_form):
                continue
            
            # Store (keep first occurrence)
            if short_form not in self.acronym_mappings:
                self.acronym_mappings[short_form] = long_form
                print(f"    ✓ {short_form} = {long_form}")
            
            self.acronym_frequencies[short_form] += 1
        
        # Find standalone acronyms (not in mappings)
        # Also filter out acronyms from heading-like contexts
        all_acronyms = re.findall(r'\b[A-Z]{2,10}\b', text)
        for acronym in all_acronyms:
            if self.is_valid_acronym(acronym):
                if acronym not in self.acronym_mappings:
                    self.standalone_acronyms[acronym] += 1
    
    def save_results(self):
        """Save results to CSV"""
        # Acronym mappings
        mapping_data = []
        for acronym in sorted(self.acronym_mappings.keys()):
            mapping_data.append({
                'acronym': acronym,
                'full_form': self.acronym_mappings[acronym],
                'frequency': self.acronym_frequencies[acronym]
            })
        
        df_mappings = pd.DataFrame(mapping_data)
        df_mappings.to_csv(self.mapping_csv, index=False)
        print(f"\n✓ Saved {len(mapping_data)} mappings to {self.mapping_csv}")
        
        # Standalone acronyms
        standalone_data = []
        for acronym in sorted(self.standalone_acronyms.keys()):
            standalone_data.append({
                'acronym': acronym,
                'frequency': self.standalone_acronyms[acronym]
            })
        
        df_standalone = pd.DataFrame(standalone_data)
        df_standalone.to_csv(self.standalone_csv, index=False)
        print(f"✓ Saved {len(standalone_data)} standalone acronyms to {self.standalone_csv}")
    
    def process_pdf(self, pdf_path):
        """Process a single PDF"""
        pdf_name = os.path.basename(pdf_path)
        print(f"\n{'='*60}")
        print(f"Processing: {pdf_name}")
        print(f"{'='*60}")
        
        # Extract text
        text = self.extract_text_from_pdf(pdf_path)
        
        if not text:
            print("No text extracted!")
            return
        
        print(f"Extracted {len(text)} characters")
        
        # Extract acronyms (headings removed inside)
        self.extract_acronyms(text)
        
        # Save results
        self.save_results()
        
        print(f"\n{'='*60}")
        print(f"Results for {pdf_name}")
        print(f"{'='*60}")
        print(f"Acronym mappings: {len(self.acronym_mappings)}")
        print(f"Standalone acronyms: {len(self.standalone_acronyms)}")
        print(f"{'='*60}\n")
        
        # Delete PDF
        try:
            safe_delete(pdf_path)
        except:
            pass
    
    def process_all_pdfs(self):
        """Process all PDFs in folder"""
        try:
            pdf_files = [
                os.path.join(self.input_folder, f)
                for f in os.listdir(self.input_folder)
                if f.lower().endswith('.pdf')
            ]
        except Exception as e:
            print(f"Error reading folder: {e}")
            return
        
        if not pdf_files:
            print("No PDF files found!")
            return
        
        print(f"\n{'='*60}")
        print(f"LIGHTWEIGHT SCHWARTZ-HEARST EXTRACTOR")
        print(f"(With Heading Removal)")
        print(f"{'='*60}")
        print(f"Found {len(pdf_files)} PDF file(s)")
        print(f"Output folder: {self.output_folder}")
        print(f"{'='*60}")
        
        for pdf_path in pdf_files:
            try:
                self.process_pdf(pdf_path)
            except Exception as e:
                print(f"Error processing {os.path.basename(pdf_path)}: {e}")
                import traceback
                traceback.print_exc()
        
        print(f"\n{'='*60}")
        print("PROCESSING COMPLETE")
        print(f"{'='*60}")
        print(f"Total mappings: {len(self.acronym_mappings)}")
        print(f"Total standalone: {len(self.standalone_acronyms)}")
        print(f"{'='*60}\n")


if __name__ == "__main__":
    INPUT_FOLDER = "digitalized_documents"
    OUTPUT_FOLDER = "output_lightweight"
    
    extractor = SchwartzHearstExtractor(INPUT_FOLDER, OUTPUT_FOLDER)
    extractor.process_all_pdfs()
