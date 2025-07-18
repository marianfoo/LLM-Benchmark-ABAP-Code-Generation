CLASS z_humaneval_021 DEFINITION
  PUBLIC
  FINAL
  CREATE PUBLIC.

  PUBLIC SECTION.
    TYPES: BEGIN OF ty_number,
             value TYPE f,
           END OF ty_number.
    TYPES: tt_numbers TYPE STANDARD TABLE OF ty_number.
    TYPES: tt_result TYPE STANDARD TABLE OF f WITH NON-UNIQUE DEFAULT KEY.

    CLASS-METHODS rescale_to_unit
      IMPORTING
        VALUE(numbers) TYPE tt_numbers
      RETURNING
        VALUE(result)  TYPE tt_result.

ENDCLASS.

CLASS z_humaneval_021 IMPLEMENTATION.

  METHOD rescale_to_unit.
    DATA: min_number TYPE f,
          max_number TYPE f.

    IF lines( numbers ) = 0.
      RETURN.
    ENDIF.

    min_number = REDUCE f( INIT min = numbers[ 1 ]-value
                          FOR num IN numbers
                          NEXT min = COND #( WHEN num-value < min THEN num-value ELSE min ) ).

    max_number = REDUCE f( INIT max = numbers[ 1 ]-value
                          FOR num IN numbers
                          NEXT max = COND #( WHEN num-value > max THEN num-value ELSE max ) ).

    IF max_number = min_number.
      LOOP AT numbers INTO DATA(num_entry).
        APPEND 0 TO result.
      ENDLOOP.
      RETURN.
    ENDIF.

    LOOP AT numbers INTO num_entry.
      DATA(rescaled_value) = ( num_entry-value - min_number ) / ( max_number - min_number ).
      APPEND rescaled_value TO result.
    ENDLOOP.

  ENDMETHOD.

ENDCLASS.