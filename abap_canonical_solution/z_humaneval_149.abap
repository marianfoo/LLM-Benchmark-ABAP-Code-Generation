CLASS z_humaneval_149 DEFINITION
  PUBLIC
  CREATE PUBLIC .

  PUBLIC SECTION.
    CLASS-METHODS: sorted_list_sum
      IMPORTING
        lst           TYPE string_table
      RETURNING
        VALUE(result) TYPE string_table.

  PRIVATE SECTION.

ENDCLASS.

CLASS z_humaneval_149 IMPLEMENTATION.

  METHOD sorted_list_sum.
    DATA: sorted_list   TYPE string_table,
          filtered_list TYPE string_table,
          temp_string   TYPE string,
          len           TYPE i.

    sorted_list = lst.
    SORT sorted_list.

    LOOP AT sorted_list INTO temp_string.
      len = strlen( temp_string ).
      IF len MOD 2 = 0.
        APPEND temp_string TO filtered_list.
      ENDIF.
    ENDLOOP.

    TYPES: BEGIN OF length_string,
             length TYPE i,
             string TYPE string,
           END OF length_string.

    DATA: sort_table TYPE TABLE OF length_string,
          sort_entry TYPE length_string.

    LOOP AT filtered_list INTO temp_string.
      sort_entry-length = strlen( temp_string ).
      sort_entry-string = temp_string.
      APPEND sort_entry TO sort_table.
    ENDLOOP.

    SORT sort_table BY length ASCENDING string ASCENDING.

    CLEAR filtered_list.
    LOOP AT sort_table INTO sort_entry.
      APPEND sort_entry-string TO filtered_list.
    ENDLOOP.

    result = filtered_list.

  ENDMETHOD.

ENDCLASS.