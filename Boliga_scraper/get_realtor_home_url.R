# This function gets the URL to the realtor website for each home listed in Boliga

# name of css element that has the realtor website URL: .px-1.ng-star-inserted
# The argument passed to the function has to be the 
# URL of the boliga web page for a specific home, e.g.:
# https://www.boliga.dk/bolig/1630201/vesterbrogade_192__5_23_1800_frederiksberg_c 
get_realtor_home_url <- function(url) {
  csselement_realtor_url <- ".px-1.ng-star-inserted" # c(".px-1.ng-star-inserted", ".font-weight-bolder.ng-star-inserted")
  html_content <- read_html(url)
  realtor_url <- html_nodes(html_content, csselement_realtor_url) %>%
    html_attr("href")
  return(realtor_url)
}

# this is the old version I used before wrapping get_realtor_home_url in possibly()
#get_realtor_home_url <- function(url) {
#  #vec <- c()
#  if (http_error(url)) {
#    return("http_error")
#  } else {
##    csselement_realtor_url <- ".px-1.ng-star-inserted" # c(".px-1.ng-star-inserted", ".font-weight-bolder.ng-star-inserted")
#    html_content <- read_html(url)
#    realtor_url <- html_nodes(html_content, csselement_realtor_url) %>%
#      html_attr("href")
#    return(realtor_url)
#  }
  
  #for (css_element in csselement_realtor_url) {
  #  realtor_url <- html_nodes(html_content, css_element) %>%
  #    html_attr("href")
  #  vec <- append(vec, realtor_url)
  #}
  
  #print(realtor_url)
  #vec
#}
