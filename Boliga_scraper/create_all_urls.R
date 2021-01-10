## This function retrieves the number of results and 
# computes the number of pages for a query
# a query is represented by a base_url
# there are 3 base urls:
# One for FRB
# One for KBH
# One for kastrup
get_number_of_pages <- function(base_url){
  # This is the name of the css element that contains the number of search results
  csselement_results <- ".flex-shrink-1" 
  # Creating a data object to store the value of the css element, 
  # ie the number of search reesults
  search_result_html <- read_html(base_url)
  # Extracting the search results from the data object
  results <- html_nodes(search_result_html, csselement_results)
  # Trimming the search results such that it's a string: "X results"
  results <- html_text(results, trim = TRUE) #trim_element(results)
  # Removing the dot in number
  results <- gsub(".","",results, fixed = TRUE)
  # Splitting the string and converting it to an integer
  results <- strsplit(results, " ")
  results <- results[[1]]
  results <- as.numeric(results[1])
  # dividing by 50 because Boliga displays 50 results pr page and rounding up
  pages <- ceiling(results/50)
  return(pages)
}


## This function  and returns a vector containing URLs for each page of a search query
# It takes a base URL as an argument
# A base URL is the URL that displays the first page a Boliga search query
# The returned vector contains URLs for each page of a search query
# except for the first page as you already have this in the base URL
# Illustrative example:
# Say you search for homes in Frederiksberg. 
# Such a query may return, say, 6 pages of results
# This function creates the URL for each of these pages
create_search_result_page_urls_vector <- function(base_url){
  url_page_part <- "&page=" # Example: url for page 2 of kbh = https://www.boliga.dk/resultat?municipality=101&page=2
  #creating a list to contain the url for each page
  desired_length_of_vector <- get_number_of_pages(base_url)
  vector_of_page_urls <- c()
  page_counter <- 2
  while (page_counter <= desired_length_of_vector){
    search_result_url <- paste(base_url, url_page_part,
                               page_counter, sep = "", collapse = NULL)
    vector_of_page_urls <- append(vector_of_page_urls, search_result_url)
    page_counter <- page_counter + 1
    print("Page counter is: ")
    print(page_counter)
  }
  return(vector_of_page_urls)
} # end create_search_result_page_urls

## This function returns a vector of all the URLs that need to be scraped
# It does not take any arguments
create_all_urls <- function(){
  vector_base_url_frb <- c("https://www.boliga.dk/resultat?municipality=147")
  vector_base_url_kbh <- c("https://www.boliga.dk/resultat?municipality=101")
  vector_base_url_kastrup <- c("https://www.boliga.dk/resultat?zipCodes=2770") # https://www.boliga.dk/resultat?zipCodes=2770
  
  search_result_urls_frb <- create_search_result_page_urls_vector(vector_base_url_frb)
  search_result_urls_kbh <- create_search_result_page_urls_vector(vector_base_url_kbh)
  search_result_urls_kastrup <- create_search_result_page_urls_vector(vector_base_url_kastrup)
  all_urls <- c(vector_base_url_frb,search_result_urls_frb,
                vector_base_url_kbh, search_result_urls_kbh,
                vector_base_url_kastrup, search_result_urls_kastrup)
  return(all_urls)
} # end create_all_urls
