CLASS z_humaneval_116 DEFINITION
  PUBLIC
  FINAL
  CREATE PUBLIC .

  PUBLIC SECTION.
    CLASS-METHODS: sort_array
      IMPORTING
        arr           TYPE int4_table
      RETURNING
        VALUE(result) TYPE int4_table.

  PRIVATE SECTION.
    CLASS-METHODS: count_ones
      IMPORTING
        number       TYPE i
      RETURNING
        VALUE(count) TYPE i.

ENDCLASS.

CLASS z_humaneval_116 IMPLEMENTATION.

  METHOD sort_array.
    TYPES: BEGIN OF ty_sort_entry,
             value      TYPE i,
             ones_count TYPE i,
           END OF ty_sort_entry.

    DATA: sort_table TYPE TABLE OF ty_sort_entry,
          sort_entry TYPE ty_sort_entry,
          temp_value TYPE i.

    LOOP AT arr INTO temp_value.
      sort_entry-value = temp_value.
      sort_entry-ones_count = count_ones( temp_value ).
      APPEND sort_entry TO sort_table.
    ENDLOOP.

    SORT sort_table BY ones_count ASCENDING value ASCENDING.

    LOOP AT sort_table INTO sort_entry.
      APPEND sort_entry-value TO result.
    ENDLOOP.
  ENDMETHOD.

  METHOD count_ones.
    DATA: temp_number TYPE i,
          remainder   TYPE i.

    temp_number = abs( number ).
    count = 0.

    WHILE temp_number > 0.
      remainder = temp_number MOD 2.
      IF remainder = 1.
        count = count + 1.
      ENDIF.
      temp_number = temp_number DIV 2.
    ENDWHILE.
  ENDMETHOD.

ENDCLASS.