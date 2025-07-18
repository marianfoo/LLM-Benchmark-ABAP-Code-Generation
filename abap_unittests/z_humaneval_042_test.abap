CLASS ltc_test_incr_list DEFINITION FOR TESTING DURATION SHORT RISK LEVEL HARMLESS.
  PRIVATE SECTION.
    METHODS: test_empty_list FOR TESTING,
      test_three_elements FOR TESTING,
      test_nine_elements FOR TESTING.
ENDCLASS.

CLASS ltc_test_incr_list IMPLEMENTATION.
  METHOD test_empty_list.
    DATA: input    TYPE TABLE OF i,
          expected TYPE TABLE OF i,
          result   TYPE TABLE OF i.

    result = z_humaneval_042=>incr_list( input ).
    cl_abap_unit_assert=>assert_equals( act = result exp = expected ).
  ENDMETHOD.

  METHOD test_three_elements.
    DATA: input    TYPE TABLE OF i,
          expected TYPE TABLE OF i,
          result   TYPE TABLE OF i.

    input = VALUE #( ( 3 ) ( 2 ) ( 1 ) ).
    expected = VALUE #( ( 4 ) ( 3 ) ( 2 ) ).

    result = z_humaneval_042=>incr_list( input ).
    cl_abap_unit_assert=>assert_equals( act = result exp = expected ).
  ENDMETHOD.

  METHOD test_nine_elements.
    DATA: input    TYPE TABLE OF i,
          expected TYPE TABLE OF i,
          result   TYPE TABLE OF i.

    input = VALUE #( ( 5 ) ( 2 ) ( 5 ) ( 2 ) ( 3 ) ( 3 ) ( 9 ) ( 0 ) ( 123 ) ).
    expected = VALUE #( ( 6 ) ( 3 ) ( 6 ) ( 3 ) ( 4 ) ( 4 ) ( 10 ) ( 1 ) ( 124 ) ).

    result = z_humaneval_042=>incr_list( input ).
    cl_abap_unit_assert=>assert_equals( act = result exp = expected ).
  ENDMETHOD.
ENDCLASS.