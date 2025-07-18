CLASS z_humaneval_005 DEFINITION
  PUBLIC
  FINAL
  CREATE PUBLIC .

  PUBLIC SECTION.
    CLASS-METHODS: intersperse
      IMPORTING
        numbers       TYPE int4_table
        delimeter     TYPE i
      RETURNING
        VALUE(result) TYPE int4_table.

ENDCLASS.

CLASS z_humaneval_005 IMPLEMENTATION.

  METHOD intersperse.
    IF lines( numbers ) = 0.
      RETURN.
    ENDIF.

    DATA: temp_numbers TYPE int4_table.
    temp_numbers = numbers.

    LOOP AT temp_numbers INTO DATA(number) FROM 1 TO lines( temp_numbers ) - 1.
      result = VALUE #( BASE result ( number ) ( delimeter ) ).
    ENDLOOP.

    READ TABLE temp_numbers INTO DATA(last_number) INDEX lines( temp_numbers ).
    result = VALUE #( BASE result ( last_number ) ).
  ENDMETHOD.

ENDCLASS.