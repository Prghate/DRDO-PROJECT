# dict_final_v20_no_mapping.py — Remove Acronym Mapping, Keep Lowercase + Standalone Acronyms

import os
import pandas as pd
import fitz  # PyMuPDF
import re
import time
import stat
from nltk.corpus import stopwords
import nltk

nltk.download('stopwords')


def safe_delete(file_path, retries=5, delay=1):
    """Try deleting a file with retries to handle file locks or permission issues."""
    for attempt in range(retries):
        try:
            if os.path.exists(file_path):
                os.chmod(file_path, stat.S_IWRITE)  # Ensure writable
                os.remove(file_path)
                print(f"✓ Deleted: {os.path.basename(file_path)}")
            break
        except PermissionError:
            print(f"⚠ File locked, retrying delete... ({attempt+1}/{retries})")
            time.sleep(delay)
        except Exception as e:
            print(f"✗ Error deleting {os.path.basename(file_path)}: {e}")
            break


class NavalDocumentProcessor:
    def __init__(self, input_folder, output_folder="output_no_mapping"):
        self.input_folder = input_folder
        self.output_folder = output_folder
        os.makedirs(self.output_folder, exist_ok=True)

        self.stop_words = set(stopwords.words('english'))

        # TWO CSV files (removed mapping file)
        self.lowercase_csv = os.path.join(self.output_folder, "lowercase_words_alphabetical.csv")
        self.standalone_acronyms_csv = os.path.join(self.output_folder, "standalone_acronyms_alphabetical.csv")
        self.log_file = os.path.join(self.output_folder, "processing_log.txt")

        self.naval_words = self._initialize_naval_vocabulary()
        
        # Track seen words (removed mapping storage)
        self.lowercase_seen = {}
        self.standalone_acronyms_seen = {}

        self.roman_pattern = re.compile(
            r'^(?=[MDCLXVI])M{0,3}(CM|CD|D?C{0,3})'
            r'(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})$', re.IGNORECASE
        )
        
        # Common mixed-case acronyms
        self.known_mixed_acronyms = {
            'phd', 'mba', 'latex', 'mysql', 'postgresql', 'javascript',
            'iphone', 'ipad', 'ipod', 'macbook', 'ios', 'macos',
            'ebay', 'etsy', 'paypal', 'linkedin', 'youtube',
            'mphil', 'btech', 'mtech'
        }
        
        # Heading keywords
        self.heading_keywords = [
            'chapter', 'section', 'part', 'article', 'annexure', 
            'appendix', 'volume', 'abstract', 'introduction', 
            'conclusion', 'summary', 'references', 'index', 'contents'
        ]

    def _initialize_naval_vocabulary(self):
        naval_terms = {
            'ship', 'vessel', 'boat', 'submarine', 'carrier', 'destroyer', 'cruiser', 
            'frigate', 'corvette', 'battleship', 'gunboat', 'patrol', 'tanker', 
            'warship', 'fleet', 'armada', 'flotilla', 'squadron', 'craft',
            'admiral', 'captain', 'commander', 'lieutenant', 'ensign', 'officer', 
            'sailor', 'seaman', 'crew', 'personnel', 'naval', 'navy', 'marine',
            'commodore', 'midshipman', 'petty', 'chief', 'warrant', 'rating',
            'bow', 'stern', 'port', 'starboard', 'deck', 'hull', 'keel', 'mast',
            'sail', 'anchor', 'rudder', 'propeller', 'bridge', 'cabin', 'hold',
            'radar', 'sonar', 'periscope', 'torpedo', 'missile', 'gun', 'cannon',
            'turret', 'hatch', 'bulkhead', 'compartment', 'engine', 'boiler',
            'galley', 'quarters', 'magazine', 'arsenal', 'armory', 'navigation',
            'voyage', 'cruise', 'patrol', 'reconnaissance', 'surveillance', 'deployment',
            'mission', 'operation', 'exercise', 'drill', 'maneuver', 'combat', 'battle',
            'engagement', 'attack', 'defense', 'blockade', 'convoy', 'escort',
            'intercept', 'pursuit', 'evasion', 'strategy', 'tactical', 'offensive',
            'defensive', 'amphibious', 'maritime', 'marine', 'nautical', 'sea', 'ocean',
            'water', 'wave', 'tide', 'current', 'depth', 'fathom', 'knot', 'league',
            'latitude', 'longitude', 'bearing', 'course', 'heading', 'position',
            'waypoint', 'route', 'channel', 'strait', 'bay', 'gulf', 'armament',
            'ammunition', 'ordnance', 'weapon', 'defense', 'system', 'communication',
            'radio', 'signal', 'flag', 'code', 'cipher', 'chart', 'map', 'compass',
            'sextant', 'chronometer', 'binoculars', 'telescope', 'rangefinder',
            'gyroscope', 'autopilot', 'supply', 'provision', 'fuel', 'stores',
            'cargo', 'logistics', 'equipment', 'maintenance', 'repair', 'overhaul',
            'refit', 'dock', 'dockyard', 'shipyard', 'base', 'port', 'harbor',
            'berth', 'mooring', 'wharf', 'pier', 'jetty', 'anchorage', 'command',
            'order', 'instruction', 'directive', 'report', 'log', 'record', 'document',
            'file', 'dispatch', 'message', 'protocol', 'procedure', 'regulation',
            'standard', 'guideline', 'manual', 'handbook', 'orders', 'watch',
            'duty', 'shift', 'hour', 'bell', 'chronometer', 'time', 'date', 'period',
            'duration', 'interval', 'schedule', 'weather', 'storm', 'gale', 'wind',
            'fog', 'visibility', 'condition', 'forecast', 'barometer', 'pressure',
            'temperature', 'climate', 'squall', 'hurricane', 'typhoon', 'cyclone',
            'monsoon', 'force', 'power', 'strength', 'capability', 'readiness',
            'status', 'operational', 'strategic', 'military', 'security', 'intelligence',
            'classified', 'confidential', 'secret', 'restricted', 'archive', 'register',
            'inventory', 'list', 'roster', 'manifest', 'plan', 'organization',
            'structure', 'hierarchy', 'chain', 'authority', 'responsibility',
            'accountability', 'control', 'speed', 'velocity', 'acceleration',
            'distance', 'range', 'radius', 'capacity', 'displacement', 'tonnage',
            'draft', 'beam', 'length', 'width', 'height', 'dimension', 'specification',
            'parameter', 'caliber', 'gauge', 'diameter', 'weight', 'payload', 'active',
            'inactive', 'deployed', 'stationed', 'assigned', 'attached', 'detached',
            'commissioned', 'decommissioned', 'ready', 'available', 'engaged', 'underway',
            'anchored', 'moored', 'berthed', 'docked', 'sailing', 'steaming', 'cruising',
            'administration', 'office', 'department', 'division', 'section', 'unit',
            'branch', 'service', 'bureau', 'directorate', 'headquarters', 'station',
            'facility', 'installation', 'establishment', 'post', 'depot', 'training',
            'qualification', 'certification', 'proficiency', 'skill', 'competence',
            'expertise', 'experience', 'knowledge', 'instruction', 'education',
            'course', 'program', 'curriculum', 'syllabus', 'academy', 'school', 'college',
            'emergency', 'alarm', 'alert', 'warning', 'danger', 'hazard', 'safety',
            'protection', 'rescue', 'evacuation', 'distress', 'damage', 'firefighting',
            'flooding', 'casualty', 'survivor', 'lifeboat', 'lifejacket', 'raft',
            'beacon', 'pennant', 'colors', 'salute', 'ceremony', 'tradition', 'custom',
            'etiquette', 'honor', 'service'
        }
        return naval_terms

    def _is_acronym(self, word):
        """Enhanced acronym detection"""
        if word.lower() in self.known_mixed_acronyms:
            return True
        
        if word.isupper() and len(word) >= 2:
            return True
        
        upper_count = sum(1 for c in word if c.isupper())
        
        if upper_count >= 2 and len(word) <= 6:
            return True
        
        if len(word) >= 3 and upper_count >= 2:
            has_consecutive_caps = False
            for i in range(len(word) - 1):
                if word[i].isupper() and word[i+1].isupper():
                    has_consecutive_caps = True
                    break
            if has_consecutive_caps:
                return True
        
        return False

    def is_heading(self, text_line):
        """Pattern-based heading detection"""
        if not text_line or len(text_line.strip()) == 0:
            return False
        
        text_line = text_line.strip()
        words = text_line.split()
        
        if len(words) == 0:
            return False
        
        # Rule 1: All caps line
        if text_line.isupper() and len(words) <= 12:
            return True
        
        # Rule 2: Short line with mostly capitals (2-6 words)
        if 2 <= len(words) <= 6:
            title_words = sum(1 for w in words if w and len(w) > 0 and w[0].isupper())
            if title_words / len(words) >= 0.7:
                return True
        
        # Rule 3: Contains heading keywords
        text_lower = text_line.lower()
        for keyword in self.heading_keywords:
            if text_lower.startswith(keyword):
                return True
        
        # Rule 4: Ends with colon
        if text_line.endswith(':'):
            return True
        
        # Rule 5: Starts with numbering
        if re.match(r'^(\d+\.|\d+\.\d+|[IVXLCDM]+\.)\s+', text_line):
            return True
        
        return False

    def extract_text_from_pdf(self, pdf_path):
        """Extract text for digitalized documents"""
        try:
            doc = fitz.open(pdf_path)
            num_pages = len(doc)
            
            all_lines = []
            
            for page_num in range(num_pages):
                page = doc.load_page(page_num)
                text = page.get_text("text")
                
                lines = text.split('\n')
                
                for line in lines:
                    line = line.strip()
                    if line:
                        all_lines.append(line)
            
            doc.close()
            return all_lines, num_pages
            
        except Exception as e:
            print(f"Error extracting text: {e}")
            return [], 0

    def extract_words_from_text(self, lines):
        """
        Extract words and standalone acronyms (NO mapping extraction):
        - Remove parenthetical content entirely
        - Standalone acronyms → Add to standalone_acronyms file
        - Regular words → Add to lowercase file
        """
        
        heading_lines_skipped = 0
        lowercase_words = []
        
        for line in lines:
            if self.is_heading(line):
                heading_lines_skipped += 1
                continue
            
            # Remove ALL parenthetical content (including acronyms and their definitions)
            cleaned_line = re.sub(r'\s*\([^)]*\)', '', line)
            
            # Extract words
            words = re.findall(r'\b[a-zA-Z]{2,}\b', cleaned_line)
            
            for word in words:
                if not word.isalpha() or not word.isascii() or self.roman_pattern.match(word):
                    continue
                
                if word.lower() in self.stop_words:
                    continue
                
                # Check if it's an acronym
                if self._is_acronym(word):
                    # Track standalone acronym
                    if word not in self.standalone_acronyms_seen:
                        self.standalone_acronyms_seen[word] = 0
                    self.standalone_acronyms_seen[word] += 1
                else:
                    # Regular word → lowercase
                    lowercase_words.append(word.lower())
        
        print(f"✓ Skipped {heading_lines_skipped} heading lines")
        print(f"✓ Found {len(self.standalone_acronyms_seen)} unique standalone acronyms")
        
        return lowercase_words

    def update_word_data(self, lowercase_words):
        """Only track lowercase words"""
        for word in lowercase_words:
            if word not in self.lowercase_seen:
                self.lowercase_seen[word] = 0
            self.lowercase_seen[word] += 1

    def save_to_csv(self):
        # 1. Save lowercase words ALPHABETICALLY
        lowercase_list = []
        for word in sorted(self.lowercase_seen.keys()):
            lowercase_list.append({
                'word': word, 
                'frequency': self.lowercase_seen[word]
            })
        pd.DataFrame(lowercase_list).to_csv(self.lowercase_csv, index=False)
        
        # 2. Save standalone acronyms ALPHABETICALLY
        standalone_list = []
        for acronym in sorted(self.standalone_acronyms_seen.keys()):
            standalone_list.append({
                'acronym': acronym,
                'frequency': self.standalone_acronyms_seen[acronym]
            })
        pd.DataFrame(standalone_list).to_csv(self.standalone_acronyms_csv, index=False)

    def log_document_results(self, doc_name, pages, lowercase_count, standalone_count):
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(f"\n{'='*60}\n")
            f.write(f"Document: {doc_name}\n")
            f.write(f"Processing Time: {pd.Timestamp.now()}\n")
            f.write(f"Pages: {pages}\n")
            f.write(f"Lowercase Words: {lowercase_count}\n")
            f.write(f"Standalone Acronyms: {standalone_count}\n")
            f.write(f"{'='*60}\n")

    def process_pdf(self, pdf_path):
        pdf_name = os.path.basename(pdf_path)
        print(f"\n{'='*60}")
        print(f"Processing: {pdf_name}")
        print(f"{'='*60}")
        
        lines, num_pages = self.extract_text_from_pdf(pdf_path)
        
        if not lines:
            print(f"✗ No text extracted from {pdf_name}")
            return
        
        print(f"✓ Extracted text from {num_pages} pages")
        
        lowercase_words = self.extract_words_from_text(lines)
        
        if not lowercase_words and not self.standalone_acronyms_seen:
            print(f"✗ No valid words found in {pdf_name}")
            return
        
        self.update_word_data(lowercase_words)
        self.save_to_csv()
        
        print(f"\n{'='*60}")
        print(f"FINAL RESULTS FOR: {pdf_name}")
        print(f"{'='*60}")
        print(f"Pages: {num_pages}")
        print(f"Lowercase Words Extracted: {len(lowercase_words)}")
        print(f"Total Unique Lowercase: {len(self.lowercase_seen)}")
        print(f"Total Standalone Acronyms: {len(self.standalone_acronyms_seen)}")
        
        self.log_document_results(pdf_name, num_pages, len(lowercase_words), len(self.standalone_acronyms_seen))
        
        try:
            safe_delete(pdf_path)
        except Exception as e:
            print(f"✗ Error deleting document: {e}")
        print(f"{'='*60}\n")

    def process_all_documents(self):
        pdf_extension = '.pdf'
        try:
            all_files = [
                os.path.join(self.input_folder, f)
                for f in os.listdir(self.input_folder)
                if os.path.isfile(os.path.join(self.input_folder, f)) and
                os.path.splitext(f)[1].lower() == pdf_extension
            ]
        except Exception as e:
            print(f"Error accessing input folder: {e}")
            return
        if not all_files:
            print("No PDF documents found in the input folder.")
            return
        print(f"\n{'='*60}")
        print(f"NAVAL DOCUMENT PROCESSOR - NO MAPPING MODE")
        print(f"{'='*60}")
        print(f"Naval Vocabulary: {len(self.naval_words)} words")
        print(f"Found {len(all_files)} PDF(s) to process")
        print(f"{'='*60}")
        successful_count = 0
        for pdf_path in all_files:
            try:
                self.process_pdf(pdf_path)
                successful_count += 1
            except Exception as e:
                print(f"Error processing {os.path.basename(pdf_path)}: {e}")
                import traceback
                traceback.print_exc()
        print(f"\n{'='*60}")
        print("PROCESSING COMPLETE")
        print(f"{'='*60}")
        print(f"PDFs Processed: {successful_count}/{len(all_files)}")
        print(f"Unique Lowercase Words: {len(self.lowercase_seen)}")
        print(f"Unique Standalone Acronyms: {len(self.standalone_acronyms_seen)}")
        print(f"\nOutput Files:")
        print(f"  1. {self.lowercase_csv}")
        print(f"  2. {self.standalone_acronyms_csv}")
        print(f"  3. {self.log_file}")
        print(f"{'='*60}\n")


# USAGE
if __name__ == "__main__":
    INPUT_FOLDER = "digitalized_documents"
    OUTPUT_FOLDER = "naval_output_no_mapping"
    processor = NavalDocumentProcessor(INPUT_FOLDER, OUTPUT_FOLDER)
    processor.process_all_documents()
