CLASS z_humaneval_146 DEFINITION
  PUBLIC
  FINAL
  CREATE PUBLIC .

  PUBLIC SECTION.
    CLASS-METHODS: special_filter
      IMPORTING
        VALUE(nums)   TYPE int4_table
      RETURNING
        VALUE(result) TYPE i.

ENDCLASS.

CLASS z_humaneval_146 IMPLEMENTATION.

  METHOD special_filter.
    DATA: count       TYPE i VALUE 0,
          num_string  TYPE string,
          first_digit TYPE i,
          last_digit  TYPE i,
          odd_digits  TYPE int4_table.

    odd_digits = VALUE #( ( 1 ) ( 3 ) ( 5 ) ( 7 ) ( 9 ) ).

    LOOP AT nums INTO DATA(num).
      IF num > 10.
        num_string = |{ abs( num ) }|.
        first_digit = substring( val = num_string off = 0 len = 1 ).
        last_digit = substring( val = num_string off = strlen( num_string ) - 1 len = 1 ).

        IF line_exists( odd_digits[ table_line = first_digit ] ) AND
           line_exists( odd_digits[ table_line = last_digit ] ).
          count = count + 1.
        ENDIF.
      ENDIF.
    ENDLOOP.

    result = count.
  ENDMETHOD.

ENDCLASS.