CLASS ltcl_test_strange_sort DEFINITION FOR TESTING DURATION SHORT RISK LEVEL HARMLESS.

  PRIVATE SECTION.
    METHODS: test_case_1 FOR TESTING,
      test_case_2 FOR TESTING,
      test_case_3 FOR TESTING,
      test_case_4 FOR TESTING,
      test_case_5 FOR TESTING,
      test_case_6 FOR TESTING,
      test_case_7 FOR TESTING,
      test_case_8 FOR TESTING,
      test_case_9 FOR TESTING.

ENDCLASS.

CLASS ltcl_test_strange_sort IMPLEMENTATION.

  METHOD test_case_1.
    DATA: input    TYPE TABLE OF i,
          expected TYPE TABLE OF i,
          actual   TYPE TABLE OF i.

    input = VALUE #( ( 1 ) ( 2 ) ( 3 ) ( 4 ) ).
    expected = VALUE #( ( 1 ) ( 4 ) ( 2 ) ( 3 ) ).
    actual = z_humaneval_070=>strange_sort_list( input ).

    cl_abap_unit_assert=>assert_equals( act = actual exp = expected ).
  ENDMETHOD.

  METHOD test_case_2.
    DATA: input    TYPE TABLE OF i,
          expected TYPE TABLE OF i,
          actual   TYPE TABLE OF i.

    input = VALUE #( ( 5 ) ( 6 ) ( 7 ) ( 8 ) ( 9 ) ).
    expected = VALUE #( ( 5 ) ( 9 ) ( 6 ) ( 8 ) ( 7 ) ).
    actual = z_humaneval_070=>strange_sort_list( input ).

    cl_abap_unit_assert=>assert_equals( act = actual exp = expected ).
  ENDMETHOD.

  METHOD test_case_3.
    DATA: input    TYPE TABLE OF i,
          expected TYPE TABLE OF i,
          actual   TYPE TABLE OF i.

    input = VALUE #( ( 1 ) ( 2 ) ( 3 ) ( 4 ) ( 5 ) ).
    expected = VALUE #( ( 1 ) ( 5 ) ( 2 ) ( 4 ) ( 3 ) ).
    actual = z_humaneval_070=>strange_sort_list( input ).

    cl_abap_unit_assert=>assert_equals( act = actual exp = expected ).
  ENDMETHOD.

  METHOD test_case_4.
    DATA: input    TYPE TABLE OF i,
          expected TYPE TABLE OF i,
          actual   TYPE TABLE OF i.

    input = VALUE #( ( 5 ) ( 6 ) ( 7 ) ( 8 ) ( 9 ) ( 1 ) ).
    expected = VALUE #( ( 1 ) ( 9 ) ( 5 ) ( 8 ) ( 6 ) ( 7 ) ).
    actual = z_humaneval_070=>strange_sort_list( input ).

    cl_abap_unit_assert=>assert_equals( act = actual exp = expected ).
  ENDMETHOD.

  METHOD test_case_5.
    DATA: input    TYPE TABLE OF i,
          expected TYPE TABLE OF i,
          actual   TYPE TABLE OF i.

    input = VALUE #( ( 5 ) ( 5 ) ( 5 ) ( 5 ) ).
    expected = VALUE #( ( 5 ) ( 5 ) ( 5 ) ( 5 ) ).
    actual = z_humaneval_070=>strange_sort_list( input ).

    cl_abap_unit_assert=>assert_equals( act = actual exp = expected ).
  ENDMETHOD.

  METHOD test_case_6.
    DATA: input    TYPE TABLE OF i,
          expected TYPE TABLE OF i,
          actual   TYPE TABLE OF i.

    input = VALUE #( ).
    expected = VALUE #( ).
    actual = z_humaneval_070=>strange_sort_list( input ).

    cl_abap_unit_assert=>assert_equals( act = actual exp = expected ).
  ENDMETHOD.

  METHOD test_case_7.
    DATA: input    TYPE TABLE OF i,
          expected TYPE TABLE OF i,
          actual   TYPE TABLE OF i.

    input = VALUE #( ( 1 ) ( 2 ) ( 3 ) ( 4 ) ( 5 ) ( 6 ) ( 7 ) ( 8 ) ).
    expected = VALUE #( ( 1 ) ( 8 ) ( 2 ) ( 7 ) ( 3 ) ( 6 ) ( 4 ) ( 5 ) ).
    actual = z_humaneval_070=>strange_sort_list( input ).

    cl_abap_unit_assert=>assert_equals( act = actual exp = expected ).
  ENDMETHOD.

  METHOD test_case_8.
    DATA: input    TYPE TABLE OF i,
          expected TYPE TABLE OF i,
          actual   TYPE TABLE OF i.

    input = VALUE #( ( 0 ) ( 2 ) ( 2 ) ( 2 ) ( 5 ) ( 5 ) ( -5 ) ( -5 ) ).
    expected = VALUE #( ( -5 ) ( 5 ) ( -5 ) ( 5 ) ( 0 ) ( 2 ) ( 2 ) ( 2 ) ).
    actual = z_humaneval_070=>strange_sort_list( input ).

    cl_abap_unit_assert=>assert_equals( act = actual exp = expected ).
  ENDMETHOD.

  METHOD test_case_9.
    DATA: input    TYPE TABLE OF i,
          expected TYPE TABLE OF i,
          actual   TYPE TABLE OF i.

    input = VALUE #( ( 111111 ) ).
    expected = VALUE #( ( 111111 ) ).
    actual = z_humaneval_070=>strange_sort_list( input ).

    cl_abap_unit_assert=>assert_equals( act = actual exp = expected ).
  ENDMETHOD.

ENDCLASS.