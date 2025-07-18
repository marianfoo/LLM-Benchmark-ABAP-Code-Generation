CLASS z_humaneval_019 DEFINITION
  PUBLIC
  FINAL
  CREATE PUBLIC.

  PUBLIC SECTION.
    CLASS-METHODS sort_numbers
      IMPORTING
        VALUE(numbers)        TYPE string
      RETURNING
        VALUE(sorted_numbers) TYPE string.

  PRIVATE SECTION.
    TYPES: BEGIN OF ty_value_map,
             word  TYPE string,
             value TYPE i,
           END OF ty_value_map.

    TYPES: tt_value_map TYPE HASHED TABLE OF ty_value_map
           WITH UNIQUE KEY word
           INITIAL SIZE 10.

    CLASS-DATA: value_map TYPE tt_value_map.

ENDCLASS.

CLASS z_humaneval_019 IMPLEMENTATION.

  METHOD sort_numbers.
    DATA: lt_numbers   TYPE TABLE OF string,
          lt_sorted    TYPE TABLE OF string,
          lt_temp      TYPE TABLE OF string,
          lv_word      TYPE string,
          ls_map_entry TYPE ty_value_map.

    " Handle empty input
    IF numbers IS INITIAL.
      sorted_numbers = ''.
      RETURN.
    ENDIF.

    " Initialize the value map if not already done
    IF lines( value_map ) = 0.
      ls_map_entry-word = 'zero'.  ls_map_entry-value = 0. INSERT ls_map_entry INTO TABLE value_map.
      ls_map_entry-word = 'one'.   ls_map_entry-value = 1. INSERT ls_map_entry INTO TABLE value_map.
      ls_map_entry-word = 'two'.   ls_map_entry-value = 2. INSERT ls_map_entry INTO TABLE value_map.
      ls_map_entry-word = 'three'. ls_map_entry-value = 3. INSERT ls_map_entry INTO TABLE value_map.
      ls_map_entry-word = 'four'.  ls_map_entry-value = 4. INSERT ls_map_entry INTO TABLE value_map.
      ls_map_entry-word = 'five'.  ls_map_entry-value = 5. INSERT ls_map_entry INTO TABLE value_map.
      ls_map_entry-word = 'six'.   ls_map_entry-value = 6. INSERT ls_map_entry INTO TABLE value_map.
      ls_map_entry-word = 'seven'. ls_map_entry-value = 7. INSERT ls_map_entry INTO TABLE value_map.
      ls_map_entry-word = 'eight'. ls_map_entry-value = 8. INSERT ls_map_entry INTO TABLE value_map.
      ls_map_entry-word = 'nine'.  ls_map_entry-value = 9. INSERT ls_map_entry INTO TABLE value_map.
    ENDIF.

    " Split the input string into words (assuming space-separated)
    SPLIT numbers AT ' ' INTO TABLE lt_temp.

    " Filter out empty strings and trim
    LOOP AT lt_temp INTO lv_word.
      CONDENSE lv_word.
      IF lv_word IS NOT INITIAL.
        APPEND lv_word TO lt_numbers.
      ENDIF.
    ENDLOOP.

    " Sort the numbers based on their numeric values
    DATA: lt_word_value TYPE TABLE OF ty_value_map,
          ls_word_value TYPE ty_value_map,
          ls_entry      TYPE ty_value_map.

    " Create helper table with word and corresponding numeric value
    LOOP AT lt_numbers INTO lv_word.
      READ TABLE value_map INTO ls_entry WITH KEY word = lv_word.
      IF sy-subrc = 0.
        ls_word_value-word = lv_word.
        ls_word_value-value = ls_entry-value.
        APPEND ls_word_value TO lt_word_value.
      ENDIF.
    ENDLOOP.

    " Sort by numeric value
    SORT lt_word_value BY value.

    " Extract sorted words
    CLEAR: lt_sorted.
    LOOP AT lt_word_value INTO ls_word_value.
      APPEND ls_word_value-word TO lt_sorted.
    ENDLOOP.

    " Join the sorted words back into a string
    LOOP AT lt_sorted INTO lv_word.
      IF sy-tabix = 1.
        sorted_numbers = lv_word.
      ELSE.
        sorted_numbers = |{ sorted_numbers } { lv_word }|.
      ENDIF.
    ENDLOOP.

  ENDMETHOD.

ENDCLASS.