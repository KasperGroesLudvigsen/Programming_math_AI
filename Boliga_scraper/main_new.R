# FRB 29-10-2020: 226 resultater
# KBH 29-10-2020: 2040 resultater
# Kastrup 29-10-2020: 115 resultater
rm(list = ls())
starttime <- Sys.time()
# 1773
library(rvest)
library(hash)
library(rlist)
library(stringr)
#library(cran)
library(sjmisc)
library(devtools)
library(httr)
library(tidyverse)

source("create_all_urls.R")
source("get_realtor_home_url.R")

all_urls <- create_all_urls()

html_content_for_all_urls_lapply <- lapply(all_urls, read_html)

df_all_home_data <- data.frame()

## UN-COMMENT THIS AFTER FIRST RUN:
df_all_home_data_from_previously <- readRDS("df_all_home_data.rds")


id_nr <- 1
css_element_all_blocks <- ".px-md-3"
csselement_address <- ".secondary-value span"
csselement_housetype <- ".text"
csselement_price <- ".d-flex.justify-content-end"
csselement_rooms <- ".d-lg-flex:nth-child(1) .text-nowrap"
csselement_size_builtyear_lotsize_expenses <- ".pr-1:nth-child(2) .text-nowrap"
csselement_floor <- ":nth-child(4) .d-md-block"

get_realtor_home_url_possibly <- possibly(get_realtor_home_url, otherwise = "error from get_realtor_home_url_possibly")
read_html_possibly <- possibly(read_html, "error_getting_floor")

# Changing the function so that it takes a single url as an argument rather than a vector of urls
get_realtor_home_description <- function(realtor_url) {
  
  csselement <- set_realtor_csselement(realtor_url)
  
  url_read_html <- read_html(realtor_url)
  
  home_description <- html_nodes(url_read_html, csselement) %>%
    html_text(trim = TRUE)
  
  home_description <- home_description[home_description != ""] %>%
    paste(collapse = ". ")
  
  return(home_description)
  
}
get_realtor_home_description_possibly <- possibly(get_realtor_home_description, "error in get_realtor_home_description")


set_realtor_csselement <- function(realtor_url) {
  if (str_contains(realtor_url, "home.dk")) return("#home-content") 
  
  if (str_contains(realtor_url, "nybolig.dk")) return(".case-facts__title , .foldable-spot__container :nth-child(1), .foldable-spot__container :nth-child(3), .foldable-spot__container :nth-child(5), p:nth-child(8)") #".foldable-spot__container"
  
  if (str_contains(realtor_url, "paulun.dk")) return("p")
  
  if (str_contains(realtor_url, "edc.dk")) return(".description p:nth-child(1) , .col-8 h1")
  
  if (str_contains(realtor_url, "estate.dk")) return(".wysiwyg p , .deck-container-above p, .property-description h2")
  
  if (str_contains(realtor_url, "lokalbolig.dk")) return(".e143rgbw1 , .e143rgbw1 p, .ewi74xi0")
  
  if (str_contains(realtor_url, "danbolig.dk")) return(".db-description-block div") 
  
  if (str_contains(realtor_url, "realmaeglerne.dk")) return(".text-full")  
  
  if (str_contains(realtor_url, "adamschnack.dk")) return(".listing-text p") 
  
  if (str_contains(realtor_url, "selvsalg.dk")) return("#js-property-description p , #js-property-description h2") 
  
  if (str_contains(realtor_url, "jespernielsen.dk")) return("p") 
  
  if (str_contains(realtor_url, "basis-bolig.dk")) return("div.column") 
  
  if (str_contains(realtor_url, "ditogmitfrb.")) return(".standard-textbox--read-more-mobile div")
  
  if (str_contains(realtor_url, "brikk.dk")) return(".prop-user-content p")
  
  if (str_contains(realtor_url, "unikboligsalg.dk")) return(".case__content p , .title")
  
  if (str_contains(realtor_url, "skbolig.dk")) return(".listing-text") 
  
  if (str_contains(realtor_url, "robinhus.dk")) return(".inner div")
  
  if (str_contains(realtor_url, "annareventlow.dk")) return("#block-6a38a1e1cdec7b288a19")
  
  if (str_contains(realtor_url, "eltoftnielsen.dk")) return("#bolig-24 .border-20")
  
  if (str_contains(realtor_url, "irvingjensen.dk")) return(".col-md-16")
  
  if (str_contains(realtor_url, "elbaeks.dk")) return("p:nth-child(8) , .expanded :nth-child(7), .expanded :nth-child(6), .expanded :nth-child(5), p:nth-child(4)") 
}

for (html_content in html_content_for_all_urls_lapply) { # this loop iterates over all the html elements
  # Creating an object that contains all html info from each block in each url
  blocks <- html_nodes(html_content, css_element_all_blocks)
  for (block in blocks) {
    
    # Creating the url for each individual home
    home_url_boliga <- html_nodes(block, "a") %>% html_attr("href")
    home_url_boliga <- paste("https://www.boliga.dk", home_url_boliga, sep = "", collapse = NULL)
    if (length(home_url_boliga) != 1) {
      for (i in home_url_boliga) {
        if (!str_contains(i, "premium")) {
          home_url_boliga <- i
          break
        }
      }
    }
    if (identical(home_url_boliga, "https://www.boliga.dk/resultat")) break
    if (home_url_boliga %in% df_all_home_data_from_previously$home_url_boliga) break
    
    print(home_url_boliga)
    
    id <- paste0("home_id_", id_nr)
    list_of_data_for_each_home <- list()
    list_of_data_for_each_home[["ID"]] <- id
    
    list_of_data_for_each_home[["home_url_boliga"]] <- home_url_boliga
    
    # Using the home_url_boliga to create the realtor url
    home_url_realtor_vector <- get_realtor_home_url_possibly(home_url_boliga)
    for (i in home_url_realtor_vector){
      if (str_contains(i, "http")) {
        home_url_realtor <- i
        break
      }
    }
    print(home_url_realtor)
    list_of_data_for_each_home[["home_url_realtor"]] <- home_url_realtor
    

    
    # Getting all the addresses
    address <- html_nodes(block, csselement_address)
    address <- html_text(address, trim = TRUE)
    list_of_data_for_each_home[["street_name_number"]] <- address[1]
    list_of_data_for_each_home[["zip_code_town"]] <- address[2]

    # Getting all the house types
    housetype <- html_nodes(block, csselement_housetype) %>% html_text(trim = TRUE)
    list_of_data_for_each_home[["Home_type"]] <- housetype[1]

    # Getting all prices
    price <- html_nodes(block, csselement_price) %>% html_text(trim = TRUE)
    price <- price[3:length(price)]
    price <- gsub(".","",price, fixed = TRUE)
    price <- word(price, -1)
    price <- str_remove_all(price, "kr")
    price <- trimws(price, which = c("both", "left","right"), whitespace = "[\\h\\v]")
    price <- strtoi(price)
    price <- price[3]
    list_of_data_for_each_home[["list_price_dkk"]] <- price

    # Getting all rooms
    rooms <- html_nodes(block, csselement_rooms) %>% html_text(trim = TRUE)
    rooms <- str_remove_all(rooms, "værelser")
    rooms <- trimws(rooms, which = c("both", "left","right"), whitespace = "[\\h\\v]")
    rooms <- strtoi(rooms)
    list_of_data_for_each_home[["rooms"]] <- rooms

    
    # Getting all sizes, built year, lotsize and expenses. 
    size_builtyear_lotsize_expenses <- html_nodes(block, csselement_size_builtyear_lotsize_expenses) %>% 
      html_text(trim = TRUE)
    
    size_m2 <- size_builtyear_lotsize_expenses[1] %>% 
      str_remove_all("m²") %>%
      trimws(which = c("both", "left","right"), whitespace = "[\\h\\v]") %>%
      strtoi()
    list_of_data_for_each_home[["home_size_m2"]] <- size_m2
  
    built_year <- size_builtyear_lotsize_expenses[2] %>% strtoi()
    list_of_data_for_each_home[["built_year"]] <- built_year

    lotsize_m2 <- size_builtyear_lotsize_expenses[3] %>%
      str_remove_all("m²") %>%
      trimws(which = c("both", "left","right"), whitespace = "[\\h\\v]") %>%
      strtoi()
    list_of_data_for_each_home[["lotsize_m2"]] <- lotsize_m2
    
    expenses <- size_builtyear_lotsize_expenses[4] %>%
      strsplit("\\s+")
    expenses <- expenses[[1]][2]
    expenses <- gsub(".","", expenses, fixed = TRUE) %>% strtoi()
    list_of_data_for_each_home[["expenses_dkk"]] <- expenses
    
    # Using home_url_realtor to fetch description
    description_of_home <- get_realtor_home_description_possibly(home_url_realtor)
    list_of_data_for_each_home[["description_of_home"]] <- description_of_home
    
    # Getting floor
    url_read_html <- read_html_possibly(home_url_boliga) #read_html(home_url_boliga)
    if (identical(url_read_html, "error_getting_floor")) {
      floor <- url_read_html
    } else {
      floor <- html_nodes(url_read_html, csselement_floor) %>%
        html_text(trim = TRUE)
    }
    
    list_of_data_for_each_home[["floor"]] <- floor
    
    # Adding the individual home home list to the df
    df_temp <- data.frame(list_of_data_for_each_home)
    df_all_home_data <- rbind(df_all_home_data, df_temp)
    
    id_nr <- id_nr + 1
    print(id_nr)
  }
}
today <- Sys.Date()
df_all_home_data[["retrieved_on"]] <- today

# UN-COMMENT THIS AFTER FIRST RUN:
df_all_home_data <- rbind(df_all_home_data_from_previously, df_all_home_data)
saveRDS(df_all_home_data, "df_all_home_data.rds")

# The csv file with the latest date is always the last updated
filename <- paste0("df_all_data_w_desc_", today, ".csv")
write_csv(df_all_home_data, filename)
       
endtime <- Sys.time()
timer_get_all_data <- endtime - starttime
print(timer_get_all_data)

