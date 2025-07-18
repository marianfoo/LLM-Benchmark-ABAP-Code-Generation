CLASS z_humaneval_012 DEFINITION
  PUBLIC
  FINAL
  CREATE PUBLIC.

  PUBLIC SECTION.

    CLASS-METHODS longest
      IMPORTING
        VALUE(strings) TYPE stringtab
      RETURNING
        VALUE(result)  TYPE string.
ENDCLASS.

CLASS z_humaneval_012 IMPLEMENTATION.
  METHOD longest.
    IF strings IS INITIAL.
      result = ''.
      RETURN.
    ENDIF.

    DATA(max_length) = 0.
    LOOP AT strings INTO DATA(current_string).
      max_length = nmax( val1 = max_length val2 = strlen( current_string ) ).
    ENDLOOP.

    LOOP AT strings INTO current_string.
      IF strlen( current_string ) = max_length.
        result = current_string.
        RETURN.
      ENDIF.
    ENDLOOP.
  ENDMETHOD.
ENDCLASS.
