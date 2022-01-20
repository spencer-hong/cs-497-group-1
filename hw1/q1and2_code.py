import re
from nltk.tokenize import word_tokenize
#from pdb import set_trace as bp
## function to use is tokenize_and_tag(path)
## accepts path, the data path to the source text
## returns list of lists, each list being a line in the source text

# returns number matches
def number_check(word):
	spans = []
	
	for match in re.finditer(r"\d+", word):
		spans.append(match)
	
	return spans

# allows 0.001 or .001 but not 4. 
def decimal_check(word):
	spans = []
	
	for match in re.finditer(r"\d*\.\d+", word):
		spans.append(match)
	
	return spans


def integer_check(word):
	spans = []
	
	for match in re.finditer(r"\d+", word):
		spans.append(match)
	
	return spans

# checks for isbn numbers
def ISBN_check(sentence):
	spans = []
	for match in re.finditer(r"(?:isbn)?\s?(?:[0-9]{3}-)?[0-9]{1,5}-[0-9]{1,7}-[0-9]{1,6}-[0-9]", sentence):
		spans.append(match)
	return spans

# checks for doi numbers
def DOI_check(sentence):
	spans = []
	for match in re.finditer(r'\b(10[.][0-9]{4,}(?:[.][0-9]+)*/(?:(?!["&\'<>])\S)+)\b', sentence):
		spans.append(match)
	
	return spans

# checks for month year or
# month, year
# checks from 1600 and onwards
def month_year_check(sentence):
	spans = []
	for match in re.finditer(r'\b(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?),?.?\s+(?:(16|17|18|19|20)\d{2})', sentence):
		spans.append(match)
	return spans

# checks for month date or 
# month, date
# i.e. jul. 2
def month_date_check(sentence):
	spans = []
	for match in re.finditer(r'\b(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?),?.?\s+?(?:\d{1,2})', sentence):
		spans.append(match)
	return spans

# checks for date month 
# i.e. 2 jul
def date_month_check(sentence):
	spans = []
	for match in re.finditer(r'(?:\d{1,2})\s+?\b(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)', sentence):
		spans.append(match)
	return spans

# dd/mm/yyyy, dd-mm-yyyy, or dd.mm.yyyy
# allows from 1600 onwards
def date_sep_check(sentence):
	spans = []
	for match in re.finditer(r'(?:(?:31(\/|-|\.)(?:0?[13578]|1[02]|(?:jan|mar|may|jul|aug|oct|dec)))\1|(?:(?:29|30)(\/|-|\.)(?:0?[1,3-9]|1[0-2]|(?:jan|mar|apr|may|jun|jul|aug|sep|oct|nov|dec))\2))(?:(?:1[6-9]|[2-9]\d)?\d{2})$|(?:29(\/|-|\.)(?:0?2|(?:feb))\3(?:(?:(?:1[6-9]|[2-9]\d)?(?:0[48]|[2468][048]|[13579][26])|(?:(?:16|[2468][048]|[3579][26])00))))$|(?:0?[1-9]|1\d|2[0-8])(\/|-|\.)(?:(?:0?[1-9]|(?:jan|feb|mar|apr|may|jun|jul|aug|sep))|(?:1[0-2]|(?:oct|nov|dec)))\4(?:(?:(16|17|18|19|20)\d{2}))', sentence):
		spans.append(match)
	
	return spans

# mm/dd/yyyy 
# allows from 1600 onwards
def month_sep_check(sentence):
	spans = []
	for match in re.finditer(r'(0[1-9]|1[0-2])\/(0[1-9]|1\d|2\d|3[01])\/(16|17|18|19|20)\d{2}', sentence):
		spans.append(match)
		
	return spans

# comma separated form: month date, year
# or month. date, year (oct. 22, 1992)
# or month. date year (oct. 22 1992)
# or month date year (oct 22 1992)
# allows from 1600 and onwards
def month_date_year_check(sentence):
	spans = []
	for match in re.finditer(r'(jan(uary)?|feb(ruary)?|mar(ch)?|apr(il)?|may|jun(e)?|jul(y)?|aug(ust)?|sep(tember)?|oct(ober)?|nov(ember)?|dec(ember)?).?\s+\d{1,2},?\s+?(?:(16|17|18|19|20)\d{2})', sentence):
		spans.append(match)
	
	return spans
# checks for standalone year mentions in text
# allows from 1800 to 2099
# i.e. John Smith was born on 1993. 
# we have to check for ISBN number FIRST before this function as it will pick up isbn numbers as well
def valid_years_in_text(sentence):
	spans = []
	for match in re.finditer(r'(18|19|20)\d{2}', sentence):
		spans.append(match)
		
	return spans


def tokenize_and_tag(path):
	'''
	accepts path that contains the source text
	'''
	

	iterator = -1
	passages = []
	with open(path, 'r', encoding = 'utf-8') as f:
		for line in f:
			
			if '<start_of_passage>' in line:
				passages.append(line.lower())
				iterator += 1
			elif '<end_of_passage>' in line:
				passages[iterator] = passages[iterator] + line.lower()
				
			else:
				passages[iterator] = passages[iterator] + line.lower()
	tokenized = []

	for passage in passages:
		if len(passage.strip()) != 0:
			moving_result = passage
			checked_for_isbn = len(ISBN_check(moving_result)) == 0

			# checking for ISBN numbers
			while not checked_for_isbn:

				match = ISBN_check(moving_result)
				match_spans = match[0].span()
				isbn_text = moving_result[match_spans[0]: match_spans[1]]

				checked_for_numbers = False

				while not checked_for_numbers:
					number_matches = number_check(isbn_text)

					number_spans = number_matches[0].span()

					isbn_text = isbn_text[:number_spans[0]] + '#' * (number_spans[1] - number_spans[0]) + isbn_text[number_spans[1]:]

					if len(number_check(isbn_text)) == 0:
						checked_for_numbers = True

				isbn_text = re.sub(r'#+', '<other>', isbn_text)

				moving_result = moving_result[:match_spans[0]] + isbn_text + moving_result[match_spans[1]:]

				if len(ISBN_check(moving_result)) == 0:
					checked_for_isbn = True
			
			# checking for DOI numbers
			
			checked_for_doi = len(DOI_check(moving_result)) == 0
			while not checked_for_doi:

				match = DOI_check(moving_result)
				match_spans = match[0].span()
				doi_text = moving_result[match_spans[0]: match_spans[1]]

				checked_for_numbers = False

				while not checked_for_numbers:
					number_matches = number_check(doi_text)

					number_spans = number_matches[0].span()

					doi_text = doi_text[:number_spans[0]] + '#' * (number_spans[1] - number_spans[0]) + doi_text[number_spans[1]:]

					if len(number_check(doi_text)) == 0:
						checked_for_numbers = True

				doi_text = re.sub(r'#+', '<other>', doi_text)

				moving_result = moving_result[:match_spans[0]] + doi_text + moving_result[match_spans[1]:]

				if len(DOI_check(moving_result)) == 0:
					checked_for_doi = True

			checked_for_month_date_year = len(month_date_year_check(moving_result)) == 0

			# checking for month_date_year_check regex
			while not checked_for_month_date_year:
				match=  month_date_year_check(moving_result)
				match_spans = match[0].span()

				date_text = moving_result[match_spans[0]: match_spans[1]]
				matches = number_check(date_text)
				if len(matches) != 0:

					# the first match is always date
					date_spans = matches[0].span()

					date_text = date_text[:date_spans[0]] + '@' * (date_spans[1] - date_spans[0]) + date_text[date_spans[1]:]

					# the next match is always year
					date_spans = matches[1].span()
					

					date_text = date_text[:date_spans[0]] + '#' * (date_spans[1] - date_spans[0]) + date_text[date_spans[1]:]

					date_text = re.sub(r'@+', '<days>', date_text)
					date_text = re.sub(r'#+', '<year>', date_text)

				moving_result = moving_result[:match_spans[0]] + date_text + moving_result[match_spans[1]:]

				if len(month_date_year_check(moving_result)) == 0:
					checked_for_month_date_year = True


			checked_for_month_sep = len(month_sep_check(moving_result)) == 0

			# checking for month_sep_check regex
			while not checked_for_month_sep:
				match=  month_sep_check(moving_result)
				match_spans = match[0].span()

				date_text = moving_result[match_spans[0]: match_spans[1]]
				matches = number_check(date_text)

				if len(matches) != 0:

					# the second match (not first) is always date (first is month)
					date_spans = matches[1].span()

					date_text = date_text[:date_spans[0]] + '@' * (date_spans[1] - date_spans[0]) + date_text[date_spans[1]:]

					# third match is always year
					date_spans = matches[2].span()

					date_text = date_text[:date_spans[0]] + '#' * (date_spans[1] - date_spans[0]) + date_text[date_spans[1]:]

					date_text = re.sub(r'@+', '<days>', date_text)
					date_text = re.sub(r'#+', '<year>', date_text)

				moving_result = moving_result[:match_spans[0]] + date_text + moving_result[match_spans[1]:]

				if len(month_sep_check(moving_result)) == 0:
					checked_for_month_sep = True


			checked_for_date_sep = len(date_sep_check(moving_result)) == 0

			# checking for date_sep_check regex
			while not checked_for_date_sep:
				match=  date_sep_check(moving_result)
				match_spans = match[0].span()

				date_text = moving_result[match_spans[0]: match_spans[1]]
				matches = number_check(date_text)

				if len(matches) != 0:

					# first match is always date
					date_spans = matches[0].span()

					date_text = date_text[:date_spans[0]] + '@' * (date_spans[1] - date_spans[0]) + date_text[date_spans[1]:]

					# third match (not second) is always year

					date_spans = matches[2].span()

					date_text = date_text[:date_spans[0]] + '#' * (date_spans[1] - date_spans[0]) + date_text[date_spans[1]:]

					date_text = re.sub(r'@+', '<days>', date_text)
					date_text = re.sub(r'#+', '<year>', date_text)

				moving_result = moving_result[:match_spans[0]] + date_text + moving_result[match_spans[1]:]

				if len(date_sep_check(moving_result)) == 0:
					checked_for_date_sep = True
			
			
			checked_for_date_month = len(date_month_check(moving_result)) == 0

			# check for month_date_check regex
			# notice that this check must be run after the month_year_check
			# as month_date_check would have picked up on month_year_check candidates first
			while not checked_for_date_month:
				match=  date_month_check(moving_result)
				match_spans = match[0].span()

				date_text = moving_result[match_spans[0]: match_spans[1]]
				matches = number_check(date_text)

				if len(matches) != 0:
					# first match is always date
					date_spans = matches[0].span()

					date_text = date_text[:date_spans[0]] + '@' * (date_spans[1] - date_spans[0]) + date_text[date_spans[1]:]

					date_text = re.sub(r'@+', '<days>', date_text)

				moving_result = moving_result[:match_spans[0]] + date_text + moving_result[match_spans[1]:]

				if len(date_month_check(moving_result)) == 0:
					checked_for_date_month = True
					
					
			checked_for_month_year = len(month_year_check(moving_result)) == 0

			# check for month_year_check regex
			while not checked_for_month_year:
				match=  month_year_check(moving_result)
				match_spans = match[0].span()

				date_text = moving_result[match_spans[0]: match_spans[1]]
				matches = number_check(date_text)

				if len(matches) != 0:

					# the only match will be year only
					date_spans = matches[0].span()

					date_text = date_text[:date_spans[0]] + '@' * (date_spans[1] - date_spans[0]) + date_text[date_spans[1]:]

					date_text = re.sub(r'@+', '<year>', date_text)

				moving_result = moving_result[:match_spans[0]] + date_text + moving_result[match_spans[1]:]

				if len(month_year_check(moving_result)) == 0:
					checked_for_month_year = True


			checked_for_month_date = len(month_date_check(moving_result)) == 0

			# check for month_date_check regex
			# notice that this check must be run after the month_year_check
			# as month_date_check would have picked up on month_year_check candidates first
			while not checked_for_month_date:
				match=  month_date_check(moving_result)
				match_spans = match[0].span()

				date_text = moving_result[match_spans[0]: match_spans[1]]
				matches = number_check(date_text)

				if len(matches) != 0:
					# first match is always date
					date_spans = matches[0].span()

					date_text = date_text[:date_spans[0]] + '@' * (date_spans[1] - date_spans[0]) + date_text[date_spans[1]:]

					date_text = re.sub(r'@+', '<days>', date_text)

				moving_result = moving_result[:match_spans[0]] + date_text + moving_result[match_spans[1]:]

				if len(month_date_check(moving_result)) == 0:
					checked_for_month_date = True


			checked_for_valid_years = len(valid_years_in_text(moving_result)) == 0

			# check for remaining numbers that fall in the valid year range
			while not checked_for_valid_years:
				match=  valid_years_in_text(moving_result)
				match_spans = match[0].span()

				date_text = moving_result[match_spans[0]: match_spans[1]]

				checked_for_numbers = len(number_check(date_text)) == 0

				while not checked_for_numbers:
					matches = number_check(date_text)

					date_spans = matches[0].span()

					date_text = date_text[:date_spans[0]] + '@' * (date_spans[1] - date_spans[0]) + date_text[date_spans[1]:]

					date_text = re.sub(r'@+', '<year>', date_text)

					if len(number_check(date_text)) == 0:
						checked_for_numbers = True

				moving_result = moving_result[:match_spans[0]] + date_text + moving_result[match_spans[1]:]



				if len(valid_years_in_text(moving_result)) == 0:
					checked_for_valid_years = True


			checked_for_decimals = len(decimal_check(moving_result)) == 0

			# check for decimals
			while not checked_for_decimals:
				match=  decimal_check(moving_result)
				match_spans = match[0].span()

				moving_result = moving_result[:match_spans[0]] + '@' * (match_spans[1] - match_spans[0]) + moving_result[match_spans[1]:]

				moving_result = re.sub(r'@+', '<decimal>', moving_result)

				if len(decimal_check(moving_result)) == 0:
					checked_for_decimals = True
			
			tokenized_words = word_tokenize(moving_result)

			for i in range(len(tokenized_words)):
				if tokenized_words[i].isnumeric():
					tokenized_words[i] = '<integer>'
				else:
					if re.search(r"\d,\d", tokenized_words[i]):
						tokenized_words[i] = '<integer>'
					else:
						if re.search(r'\d', tokenized_words[i]):
							tokenized_words[i] = '<other>'
			
			
			new_results = []
			i = 0
			while i < len(tokenized_words):
				if tokenized_words[i] == '<' and tokenized_words[i+2] == '>':

					if tokenized_words[i+1] in ['other', 'integer', 'days', 'year', 'decimal', 'end_of_passage', 'start_of_passage']:
						new_results.append('<' + tokenized_words[i+1] + '>')

						i += 3
				else:
					new_results.append(tokenized_words[i])
					i += 1

			tokenized.append(new_results)
	
	return tokenized
	
if __name__ == '__main__':
	test = tokenize_and_tag('source_text.txt')[:3]
	#bp()
	
